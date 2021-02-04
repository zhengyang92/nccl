/*************************************************************************
* Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "devcomm.h"
#include "primitives.h"
#include "collectives.h"
#include <cassert>
//#include <cooperative_groups.h>
//namespace cg = cooperative_groups;


template<class FUNC, typename T, int UNROLL>
class ncclFunction<ncclFuncAllGather, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE, FUNC, T, UNROLL> {
  public:
    __device__ void run(struct ncclWorkElem* args) {
      const int tid = threadIdx.x;
      const int nthreads = args->nThreads-WARP_SIZE;
      const int bid = blockIdx.x; //args->coll.bid;
      const int nChannels = args->coll.nChannels;
      struct ncclDevComm* comm = args->comm;
      const int nranks = comm->nRanks;
      int channelId = bid / (nranks-1);
      struct ncclChannel* channel = comm->channels + channelId;
      struct ncclRing* ring = &channel->ring;
      const int stepSize = comm->buffSizes[NCCL_PROTO_SIMPLE] / (sizeof(T)*NCCL_STEPS);
      const int chunkSize = stepSize * 1;//ALLGATHER_CHUNKSTEPS;
      const int blocksPerLink = nChannels/nranks;
      //const ssize_t loopSize = nChannels*(ssize_t)chunkSize;
      const ssize_t loopSize = chunkSize * blocksPerLink;//comm->buffSizes[NCCL_PROTO_SIMPLE] / sizeof(T);
      const ssize_t size = args->coll.count;

      int myRank = ring->devUserRanks[0];
      // Compute pointers
      const T * __restrict__ thisInput = (const T*)args->sendbuff;
      T * __restrict__ thisOutput = (T*)args->recvbuff;

      int sizePerChannel = size/nChannels;
      int nghr = (bid % (nranks-1))+1;
      int ncclWorkIndex = args->index;


      if (myRank == 0){
	ncclPrimitives<UNROLL, 1, 1, T, 0, 1, 1, FUNC>
	  prims(tid, nthreads, NULL, &nghr, thisOutput, stepSize, channel, comm, ncclShmem->ptrs, 0);
	prims.directSend(thisOutput + channelId * sizePerChannel, channelId * sizePerChannel, sizePerChannel);
      } else if (nghr == myRank) {
        int nghr = 0;
	int m1 = -1;
        ncclPrimitives<UNROLL, 1, 1, T, 1, 1, 1, FUNC>
            prims(tid, nthreads, &nghr, &m1, thisOutput, stepSize, channel, comm, ncclShmem->ptrs, 0);
        prims.directRecv(thisOutput + channelId * sizePerChannel, channelId * sizePerChannel, sizePerChannel);
      }

      //cg::grid_group barrier = cg::this_grid();
      //barrier.sync();
    }
};

#if 0
template<class FUNC, typename T, int UNROLL>
class ncclFunction<ncclFuncAllGather, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE, FUNC, T, UNROLL> {
  public:
    __device__ void run(struct ncclWorkElem* args) {
      const int tid = threadIdx.x;
      const int nthreads = args->nThreads-WARP_SIZE;
      const int bid = args->coll.bid;
      const int nChannels = args->coll.nChannels;
      struct ncclDevComm* comm = args->comm;
      struct ncclChannel* channel = comm->channels+blockIdx.x;
      struct ncclRing* ring = &channel->ring;
      const int stepSize = comm->buffSizes[NCCL_PROTO_SIMPLE] / (sizeof(T)*NCCL_STEPS);
      const int chunkSize = stepSize * ALLGATHER_CHUNKSTEPS;
      const int nranks = comm->nRanks;
      const ssize_t loopSize = nChannels*(ssize_t)chunkSize;
      const ssize_t size = args->coll.count;

      // Compute pointers
      const T * __restrict__ thisInput = (const T*)args->sendbuff;
      T * __restrict__ thisOutput = (T*)args->recvbuff;

      ncclPrimitives<UNROLL, ALLGATHER_CHUNKSTEPS/ALLGATHER_SLICESTEPS, ALLGATHER_SLICESTEPS, T, 1, 1, 1, FUNC>
        prims(tid, nthreads, &ring->prev, &ring->next, thisOutput, stepSize, channel, comm, ncclShmem->ptrs, 0);

      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        int realChunkSize = min(chunkSize, DIVUP(size-gridOffset,nChannels));
        ALIGN_SIZE(realChunkSize, nthreads*sizeof(uint64_t)/sizeof(T));
        ssize_t chunkOffset = gridOffset + bid*realChunkSize;

        /////////////// begin AllGather steps ///////////////
        ssize_t offset;
        int nelem = min(realChunkSize, size-chunkOffset);
        int rankDest;

        // step 0: push data to next GPU
        rankDest = ring->devUserRanks[0];
        offset = chunkOffset + rankDest * size;

        if (thisInput + chunkOffset == thisOutput + offset) { // In place
	  //  prims.directSend(thisInput+chunkOffset, offset, nelem);
	  prims.directSend(thisOutput + offset, offset, nelem); // 0 -x-> 1 -x-> 2
        } else {
          prims.directCopySend(thisInput+chunkOffset, thisOutput+offset, offset, nelem);
        }

        // k-2 steps: copy to next GPU
        for (int j=1; j<nranks-1; ++j) {
          rankDest = ring->devUserRanks[nranks-j];
          offset = chunkOffset + rankDest * size;

          prims.directRecvCopySend(thisOutput+offset, offset, nelem);
        }

        // Make final copy from buffer to dest.
        rankDest = ring->devUserRanks[1];
        offset = chunkOffset + rankDest * size;

        // Final wait/copy.
        prims.directRecv(thisOutput+offset, offset, nelem);
      }
    }
};
#endif

template<class FUNC, typename T, int UNROLL>
class ncclFunction<ncclFuncAllGather, NCCL_ALGO_RING, NCCL_PROTO_LL, FUNC, T, UNROLL> {
  public:
    __device__ void run(struct ncclWorkElem* args) {
      const int tid = threadIdx.x;
      const int nthreads = args->nThreads;
      const int bid = args->coll.bid;
      const int nChannels = args->coll.nChannels;
      struct ncclDevComm* comm = args->comm;
      struct ncclChannel* channel = comm->channels+blockIdx.x;
      struct ncclRing* ring = &channel->ring;
      const int stepLines = comm->buffSizes[NCCL_PROTO_LL] / (sizeof(union ncclLLFifoLine)*NCCL_STEPS);
      ssize_t chunkSize = stepLines * sizeof(uint64_t) / sizeof(T);
      const int nranks = comm->nRanks;
      const ssize_t loopSize = nChannels*chunkSize;
      const ssize_t size = args->coll.count;

      ncclLLPrimitives<T, FUNC, 1, 1> LLprims(tid, nthreads, &ring->prev, &ring->next, stepLines, channel, comm);

      // Compute pointers
      const T * __restrict__ thisInput = (const T*)args->sendbuff;
      T * __restrict__ thisOutput = (T*)args->recvbuff;

      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        if (size-gridOffset < loopSize) {
          chunkSize = args->coll.lastChunkSize;
        }
        ssize_t chunkOffset = gridOffset + bid*chunkSize;

        /////////////// begin AllGather steps ///////////////
        ssize_t offset;
        int nelem = min(chunkSize, size-chunkOffset);
        int rankDest;

        // step 0: push data to next GPU
        rankDest = ring->devUserRanks[0];
        offset = chunkOffset + rankDest * size;

        if (thisInput + chunkOffset == thisOutput + offset) { // In place
          LLprims.send(thisInput+chunkOffset, nelem);
        } else {
          LLprims.copySend(thisInput+chunkOffset, thisOutput+offset, nelem);
        }

        // k-2 steps: copy to next GPU
        for (int j=1; j<nranks-1; ++j) {
          rankDest = ring->devUserRanks[nranks-j];
          offset = chunkOffset + rankDest * size;

          LLprims.recvCopySend(thisOutput+offset, nelem);
        }

        // step k-1: final store
        rankDest = ring->devUserRanks[1];
        offset = chunkOffset + rankDest * size;

        LLprims.recv(thisOutput+offset, nelem);
      }
    }
};


#if 1
#include "prims_ll128.h"
template<class FUNC, typename T, int UNROLL>
class ncclFunction<ncclFuncAllGather, NCCL_ALGO_RING, NCCL_PROTO_LL128, FUNC, T, UNROLL> {
  public:
    __device__ void run(struct ncclWorkElem* args) {
      const int tid = threadIdx.x;
      const int nthreads = args->nThreads;
      const int bid = blockIdx.x; //args->coll.bid;
      const int nChannels = args->coll.nChannels;
      struct ncclDevComm* comm = args->comm;
      struct ncclChannel* channel = comm->channels;
      struct ncclRing* ring = &channel->ring;
      const int stepSize = comm->buffSizes[NCCL_PROTO_LL128] / (sizeof(uint64_t)*NCCL_STEPS);
      ssize_t chunkSize = stepSize*NCCL_LL128_DATAELEMS*sizeof(uint64_t) / (NCCL_LL128_LINEELEMS*sizeof(T));
      // We should not need the final /2 but it makes performance much, much smoother. Might be a bug somewhere.
      const ssize_t minChunkSize = (NCCL_LL128_SHMEM_ELEMS_PER_THREAD*nthreads*NCCL_LL128_DATAELEMS*sizeof(uint64_t))/(NCCL_LL128_LINEELEMS*sizeof(T))/2;
      const int nranks = comm->nRanks;
      const ssize_t loopSize = nChannels*chunkSize;
      const ssize_t size = args->coll.count;

      //ncclLL128Primitives<T, FUNC, 1, 1> LLprims(tid, nthreads, &ring->prev, &ring->next, stepSize, channel, comm);

      // Compute pointers
      const T * __restrict__ thisInput = (const T*)args->sendbuff;
      T * __restrict__ thisOutput = (T*)args->recvbuff;

      int myRank = ring->devUserRanks[0];

      int conn[4][3] = {
        1,2,3,
        0,2,3,
        0,1,3,
        0,1,2
      };

      // step x gpu x bid
      int schedule[1][4][6] = {
         0,  0,  0,  1,  2,  3,
         1,  1,  1,  0,  2,  3,
         2,  2,  2,  0,  1,  3,
         3,  3,  3,  0,  1,  2,
      };
      int nghr = conn[myRank][bid % 3];
      int m1 = -1;
      if (bid < 3){
        ncclLL128Primitives<T, FUNC, 1, 1>
            prims(tid, nthreads, &m1, &nghr, stepSize, comm->channels, comm);
        for (int step = 0; step < 1; step++){
          int curSchedule = schedule[step][myRank][bid];
          if (curSchedule != -1){
            prims.send(thisOutput + curSchedule * size, size);
          }
        }
      } else {
        // minus 1 is needed to pass as a "sender" to ncclPrimitive 
        // TODO: saemal, can you fix ncclPrimitive to get rid of this issue?
        ncclLL128Primitives<T, FUNC, 1, 1>
            prims(tid, nthreads, &nghr, &m1, stepSize, comm->channels, comm);
        for (int step = 0; step < 1; step++){
          int curSchedule = schedule[step][myRank][bid];
          if (curSchedule != -1){
            prims.recv(thisOutput + curSchedule * size, size);
          }
        }
      }
      return;

/*
      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        chunkSize = min(DIVUP(size-gridOffset, nChannels*minChunkSize)*minChunkSize, chunkSize);

        ssize_t chunkOffset = gridOffset + bid*chunkSize;

        /////////////// begin AllGather steps ///////////////
        ssize_t offset;
        int nelem = min(chunkSize, size-chunkOffset);
        int rankDest;

        // step 0: push data to next GPU
        rankDest = ring->devUserRanks[0];
        offset = chunkOffset + rankDest * size;

        if (thisInput + chunkOffset == thisOutput + offset) { // In place
          LLprims.send(thisInput+chunkOffset, nelem);
        } else {
          LLprims.copySend(thisInput+chunkOffset, thisOutput+offset, nelem);
        }

        // k-2 steps: copy to next GPU
        for (int j=1; j<nranks-1; ++j) {
          rankDest = ring->devUserRanks[nranks-j];
          offset = chunkOffset + rankDest * size;

          LLprims.recvCopySend(thisOutput+offset, nelem);
        }

        // step k-1: final store
        rankDest = ring->devUserRanks[1];
        offset = chunkOffset + rankDest * size;

        LLprims.recv(thisOutput+offset, nelem);
      }
      */
    }
};

#else

#include "prims_ll128.h"
template<class FUNC, typename T, int UNROLL>
class ncclFunction<ncclFuncAllGather, NCCL_ALGO_RING, NCCL_PROTO_LL128, FUNC, T, UNROLL> {
  public:
    __device__ void run(struct ncclWorkElem* args) {
      const int tid = threadIdx.x;
      const int nthreads = args->nThreads;
      const int bid = args->coll.bid;
      const int nChannels = args->coll.nChannels;
      struct ncclDevComm* comm = args->comm;
      struct ncclChannel* channel = comm->channels+blockIdx.x;
      struct ncclRing* ring = &channel->ring;
      const int stepSize = comm->buffSizes[NCCL_PROTO_LL128] / (sizeof(uint64_t)*NCCL_STEPS);
      ssize_t chunkSize = stepSize*NCCL_LL128_DATAELEMS*sizeof(uint64_t) / (NCCL_LL128_LINEELEMS*sizeof(T));
      // We should not need the final /2 but it makes performance much, much smoother. Might be a bug somewhere.
      const ssize_t minChunkSize = (NCCL_LL128_SHMEM_ELEMS_PER_THREAD*nthreads*NCCL_LL128_DATAELEMS*sizeof(uint64_t))/(NCCL_LL128_LINEELEMS*sizeof(T))/2;
      const int nranks = comm->nRanks;
      const ssize_t loopSize = nChannels*chunkSize;
      const ssize_t size = args->coll.count;

      ncclLL128Primitives<T, FUNC, 1, 1> LLprims(tid, nthreads, &ring->prev, &ring->next, stepSize, channel, comm);

      // Compute pointers
      const T * __restrict__ thisInput = (const T*)args->sendbuff;
      T * __restrict__ thisOutput = (T*)args->recvbuff;

      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        chunkSize = min(DIVUP(size-gridOffset, nChannels*minChunkSize)*minChunkSize, chunkSize);

        ssize_t chunkOffset = gridOffset + bid*chunkSize;

        /////////////// begin AllGather steps ///////////////
        ssize_t offset;
        int nelem = min(chunkSize, size-chunkOffset);
        int rankDest;

        // step 0: push data to next GPU
        rankDest = ring->devUserRanks[0];
        offset = chunkOffset + rankDest * size;

        if (thisInput + chunkOffset == thisOutput + offset) { // In place
          LLprims.send(thisInput+chunkOffset, nelem);
        } else {
          LLprims.copySend(thisInput+chunkOffset, thisOutput+offset, nelem);
        }

        // k-2 steps: copy to next GPU
        for (int j=1; j<nranks-1; ++j) {
          rankDest = ring->devUserRanks[nranks-j];
          offset = chunkOffset + rankDest * size;

          LLprims.recvCopySend(thisOutput+offset, nelem);
        }

        // step k-1: final store
        rankDest = ring->devUserRanks[1];
        offset = chunkOffset + rankDest * size;

        LLprims.recv(thisOutput+offset, nelem);
      }
    }
};
#endif

template<int PROTO, class FUNC, typename T, int UNROLL>
class ncclFunction<ncclFuncAllGather, NCCL_ALGO_TREE, PROTO, FUNC, T, UNROLL> {
  public:
    __device__ void run(struct ncclWorkElem* args) {}
};

template<int PROTO, class FUNC, typename T, int UNROLL>
class ncclFunction<ncclFuncAllGather, NCCL_ALGO_COLLNET, PROTO, FUNC, T, UNROLL> {
  public:
    __device__ void run(struct ncclWorkElem* args) {}
};

