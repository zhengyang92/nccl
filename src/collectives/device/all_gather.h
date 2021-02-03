/*************************************************************************
* Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "devcomm.h"
#include "primitives.h"
#include "collectives.h"
#include <cassert>
#include <cooperative_groups.h>
#include "sccl_all_gather.h"


template<class FUNC, typename T, int UNROLL>
class ncclFunction<ncclFuncAllGather, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE, FUNC, T, UNROLL> {
  public:
    __device__ void run(struct ncclWorkElem* args) {
      const int tid = threadIdx.x;
      const int nthreads = args->nThreads-WARP_SIZE;
      const int bid = blockIdx.x; //args->coll.bid;
      const int nChannels = args->coll.nChannels;
      struct ncclDevComm* comm = args->comm;
      struct ncclChannel* channel = comm->channels;
      struct ncclRing* ring = &channel->ring;
      const int stepSize = comm->buffSizes[NCCL_PROTO_SIMPLE] / (sizeof(T)*NCCL_STEPS);
      const int chunkSize = stepSize * 1;//ALLGATHER_CHUNKSTEPS;
      const int nranks = comm->nRanks;
      const int blocksPerLink = nChannels/nranks;
      //const ssize_t loopSize = nChannels*(ssize_t)chunkSize;
      const ssize_t loopSize = chunkSize * blocksPerLink;//comm->buffSizes[NCCL_PROTO_SIMPLE] / sizeof(T);
      const ssize_t size = args->coll.count;

      int myRank = ring->devUserRanks[0];
      // Compute pointers
      const T * __restrict__ thisInput = (const T*)args->sendbuff;
      T * __restrict__ thisOutput = (T*)args->recvbuff;
      volatile int* signals = comm->signals;
      // if (tid == 0 && bid == 0)
      //   printf("Here\n");
      /*
	 send 0 from 0 to 2 at time 0
	 send 1 from 1 to 2 at time 0
	 send 2 from 2 to 3 at time 0

	 send 0 from 2 to 1 at time 1
	 send 0 from 2 to 3 at time 1
	 send 1 from 2 to 0 at time 1
	 send 1 from 2 to 3 at time 1
	 send 2 from 3 to 0 at time 1
	 send 2 from 3 to 1 at time 1
	 send 3 from 3 to 0 at time 1
	 send 3 from 3 to 1 at time 1
	 send 3 from 3 to 2 at time 1
	 */

      // chunk 0:
      // 0 -> 2 -> 1, 3

      // chunk 1:
      // 1 -> 2 -> 0, 3

      // chunk 2:
      // 2 -> 3 -> 0, 1

      // chunk 3:
      //      3 -> 2
      // 3 -> 0, 1,
      // if (tid == 0)
      //     printf("nBlocks = %d\n", (int) gridDim.x);
      // if(nranks != 4 || (nChannels != 1)) {
	    //   printf("this is bad %d %d\n", (int) nranks, (int) nChannels);
	    //   return;
      // }

      // if (bid > 0) return;
      // ncclPrimitives<UNROLL, ALLGATHER_CHUNKSTEPS/ALLGATHER_SLICESTEPS, ALLGATHER_SLICESTEPS, T, 1, 1, 1, FUNC>
      //   prims(tid, nthreads, &ring->prev, &ring->next, thisOutput, stepSize, channel, comm, ncclShmem->ptrs, 0);
   	  // prims.directSend(thisOutput + myRank * size, myRank * size, size); // 0 -x-> 1 -x-> 2
      // prims.directRecvCopySend(thisOutput + ((myRank - 1 + 4) % 4) * size, ((myRank - 1 + 4) % 4) * size, size);
      // prims.directRecvCopySend(thisOutput + ((myRank - 2 + 4) % 4) * size, ((myRank - 2 + 4) % 4) * size, size);
      // prims.directRecv(thisOutput + ((myRank - 3 + 4) % 4) * size, ((myRank - 3 + 4) % 4) * size, size);
      // return;
/*
      int conn[4][3] = {
        1,2,3,
        0,2,3,
        0,1,3,
        0,1,2
      };

      int ncclWorkIndex = args->index;
      const int nsteps = 3;
      // step x gpu x bid
      int schedule[nsteps][4][6] = {
         0, -1, -1, -1, -1,  3,
        -1,  1, -1,  0, -1, -1,
        -1, -1,  2, -1,  1, -1,
         3, -1, -1, -1, -1,  2,

         3, -1, -1, -1, -1,  2,
        -1,  0, -1,  3, -1, -1,
        -1, -1,  1, -1,  0, -1,
         2, -1, -1, -1, -1,  1,

         2, -1, -1, -1, -1,  1,
        -1,  3, -1,  2, -1, -1,
        -1, -1,  0, -1,  3, -1,
         1, -1, -1, -1, -1,  0,
      };*/
      int ncclWorkIndex = args->index;
      int nghr = neighbors[myRank][(bid / NCHANNELS) % NNBGRS];
      if (bid / NCHANNELS < NNBGRS){
        ncclPrimitives<UNROLL, 1, 1, T, 0, 1, 1, FUNC>
            prims(tid, nthreads, NULL, &nghr, thisOutput, stepSize, comm->channels, comm, ncclShmem->ptrs, 0);
        for (int step = 0; step < NSTEPS; step++){
          int curSchedule = schedule[step][myRank][bid];
          if (curSchedule != -1){
            if (step > 0){
              while (*(signals+curSchedule) != ncclWorkIndex){}
            }
            prims.directSend(thisOutput + curSchedule * size, curSchedule * size, size);
          }
        }
      } else {
        // minus 1 is needed to pass as a "sender" to ncclPrimitive 
        // TODO: saemal, can you fix ncclPrimitive to get rid of this issue?
        int m1 = -1;
        ncclPrimitives<UNROLL, 1, 1, T, 1, 1, 1, FUNC>
            prims(tid, nthreads, &nghr, &m1, thisOutput, stepSize, comm->channels, comm, ncclShmem->ptrs, 0);
        for (int step = 0; step < NSTEPS; step++){
          int curSchedule = schedule[step][myRank][bid];
          if (curSchedule != -1){
            prims.directRecv(thisOutput + curSchedule * size, curSchedule * size, size);
            if (NSTEPS > 1 && tid == 0){
              signals[curSchedule] = ncclWorkIndex;
            }
          }
        }
      }
      // cg::grid_group barrier = cg::this_grid();
      // barrier.sync();
      // if (tid == 0 && bid == 0){
      //   for (int t = 0; t < 4; t++)
      //     signals[t] = 0;
      // }
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

