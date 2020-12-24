/*************************************************************************
* Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "devcomm.h"
#include "primitives.h"
#include "collectives.h"
#include <cassert>

#if 1
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
	// 0(0) -> 2(1) -> 1(2), 3(2)

	// chunk 1:
	// 1(0) -> 2(1) -> 0(2), 3(2)

	// chunk 2:
	// 2(0) -> 3(1) -> 0(2), 1(2)

	// chunk 3:
	// 3(1) -> 0(2), 1(2), 2(2)
      if(nranks != 4 || (nChannels != nranks)) {
      	printf("this is bad\n");
      	return;
      }
      
      int m1 = -1;
	
      if(myRank == 0) {
	      switch (bid){
		      case 0:
						// chunk 0
						{int dst0 = 2;
						ncclPrimitives<UNROLL, 1, 1, T, 1, 1, 1, FUNC>
										prims0(tid, nthreads, &m1, &dst0, thisOutput, stepSize, channel, comm, ncclShmem->ptrs, 0);
						prims0.directSend(thisInput, 0, size);}
			      break;
		      case 1:
			      // chunk 1
			      {int src1 = 2;
			      ncclPrimitives<UNROLL, 1, 1, T, 1, 1, 1, FUNC>
				      prims1(tid, nthreads, &src1, &m1, thisOutput, stepSize, channel, comm, ncclShmem->ptrs, 0);
						prims1.directRecv(thisOutput + 1 * size, 1 * size, size);}
			      break;
		      case 2:
			      // chunk 2 
			      {int src2 = 3;
			      ncclPrimitives<UNROLL, 1, 1, T, 1, 1, 1, FUNC>
				      prims2(tid, nthreads, &src2, &m1, thisOutput, stepSize, channel, comm, ncclShmem->ptrs, 0);
						prims2.directRecv(thisOutput + 2 * size, 2 * size, size);}
			      break;
		      case 3:
			      // chunk 3
			      {int src3 = 3;
			      ncclPrimitives<UNROLL, 1, 1, T, 1, 1, 1, FUNC>
				      prims3(tid, nthreads, &src3, &m1, thisOutput, stepSize, channel, comm, ncclShmem->ptrs, 0);
						prims3.directRecv(thisOutput + 3 * size, 3 * size, size);}
			      break;
          default:
            printf("Yo we should not be here\n");
            return;
	      }
      } else if (myRank == 1) {
	      switch (bid){
		      case 0:
	          // chunk 0
			      {int src0 = 2;
			      ncclPrimitives<UNROLL, 1, 1, T, 1, 1, 1, FUNC>
				      prims0(tid, nthreads, &src0, &m1, thisOutput, stepSize, channel, comm, ncclShmem->ptrs, 0);
						prims0.directRecv(thisOutput, 0, size);}
			      break;
		      case 1:
			      // chunk 1
			      {int dst1 = 2;
			      ncclPrimitives<UNROLL, 1, 1, T, 1, 1, 1, FUNC>
				      prims1(tid, nthreads, &m1, &dst1, thisOutput, stepSize, channel, comm, ncclShmem->ptrs, 0);
						prims1.directSend(thisInput, 0, size);}
			      break;
		      case 2:
			      // chunk 2 
			      {int src2 = 3;
			      ncclPrimitives<UNROLL, 1, 1, T, 1, 1, 1, FUNC>
				      prims2(tid, nthreads, &src2, &m1, thisOutput, stepSize, channel, comm, ncclShmem->ptrs, 0);
						prims2.directRecv(thisOutput + 2 * size, 2 * size, size);}
			      break;
		      case 3:
			      // chunk 3
			      {int src3 = 3;
			      ncclPrimitives<UNROLL, 1, 1, T, 1, 1, 1, FUNC>
				      prims3(tid, nthreads, &src3, &m1, thisOutput, stepSize, channel, comm, ncclShmem->ptrs, 0);
						prims3.directRecv(thisOutput + 3 * size, 3 * size, size);}
			      break;
          default:
            printf("Yo we should not be here\n");
            return;
	      }
      } else if (myRank == 2) {
	      switch (bid){
		      case 0:
	          // chunk 0
			      {int src0 = 0;
            int dst0[2] = {1,3};
			      ncclPrimitives<UNROLL, 1, 1, T, 1, 2, 1, FUNC>
				      prims0(tid, nthreads, &src0, dst0, thisOutput, stepSize, channel, comm, ncclShmem->ptrs, 0);
						prims0.directRecvCopySend(thisOutput, 0, size);}
			      break;
		      case 1:
			      // chunk 1
			      {int src1 = 1;
            int dst1[2] = {0,3};
			      ncclPrimitives<UNROLL, 1, 1, T, 1, 2, 1, FUNC>
				      prims1(tid, nthreads, &src1, dst1, thisOutput, stepSize, channel, comm, ncclShmem->ptrs, 0);
						prims1.directRecvCopySend(thisOutput + 1 * size, 1 * size, size);}
			      break;
		      case 2:
			      // chunk 2 
			      {int dst2 = 3;
			      ncclPrimitives<UNROLL, 1, 1, T, 1, 1, 1, FUNC>
				      prims2(tid, nthreads, &m1, &dst2, thisOutput, stepSize, channel, comm, ncclShmem->ptrs, 0);
						prims2.directSend(thisInput, 0, size);}
			      break;
		      case 3:
			      // chunk 3
			      {int src3 = 3;
			      ncclPrimitives<UNROLL, 1, 1, T, 1, 1, 1, FUNC>
				      prims3(tid, nthreads, &src3, &m1, thisOutput, stepSize, channel, comm, ncclShmem->ptrs, 0);
						prims3.directRecv(thisOutput + 3 * size, 3 * size, size);}
			      break;
          default:
            printf("Yo we should not be here\n");
            return;
        }
      } else if (myRank == 3) {
	      switch (bid){
		      case 0:
	          // chunk 0
			      {int src0 = 2;
			      ncclPrimitives<UNROLL, 1, 1, T, 1, 1, 1, FUNC>
				      prims0(tid, nthreads, &src0, &m1, thisOutput, stepSize, channel, comm, ncclShmem->ptrs, 0);
						prims0.directRecv(thisOutput, 0, size);}
			      break;
		      case 1:
			      // chunk 1
			      {int src1 = 2;
			      ncclPrimitives<UNROLL, 1, 1, T, 1, 1, 1, FUNC>
				      prims1(tid, nthreads, &src1, &m1, thisOutput, stepSize, channel, comm, ncclShmem->ptrs, 0);
						prims1.directRecv(thisOutput + 1 * size, 1 * size, size);}
			      break;
		      case 2:
			      // chunk 2 
            {int src2 = 2;
			      int dst2 [2] = {0,1};
			      ncclPrimitives<UNROLL, 1, 1, T, 1, 2, 1, FUNC>
				      prims2(tid, nthreads, &src2, dst2, thisOutput, stepSize, channel, comm, ncclShmem->ptrs, 0);
						prims2.directRecvCopySend(thisOutput + 2 * size, 2 * size, size);}
			      break;
		      case 3:
			      // chunk 3
			      {int dst3 [3] = {0,1,2};
			      ncclPrimitives<UNROLL, 1, 1, T, 1, 3, 1, FUNC>
				      prims3(tid, nthreads, &m1, dst3, thisOutput, stepSize, channel, comm, ncclShmem->ptrs, 0);
						prims3.directSend(thisInput, 0, size);}
			      break;
          default:
            printf("Yo we should not be here\n");
            return;
        }
      } else {
        printf("Whattttt\n");
      }
#if 0
      
/*      int conn[] = { 
      	0, 1, 2, 3, 
      	3, 0, 1, 2, 
       	2, 3, 0, 1, 
       	1, 2, 3, 0, 
      }; */

      int conn[] = {
      	0, 1,
      	1, 0
      };

      int myRank = ring->devUserRanks[0];

      if(nranks != 2 || (nChannels % nranks) != 0) {
      	printf("this is bad\n");
      	return;
      }
      
      // Compute pointers
      const T * __restrict__ thisInput = (const T*)args->sendbuff;
      T * __restrict__ thisOutput = (T*)args->recvbuff;

      int myNghr = conn[myRank * nranks + (bid / blocksPerLink)];
      if(myNghr == myRank) {
      	return;
      }

      //ncclPrimitives<UNROLL, ALLGATHER_CHUNKSTEPS/ALLGATHER_SLICESTEPS, ALLGATHER_SLICESTEPS, T, 1, 1, 0, FUNC>
      ncclPrimitives<UNROLL, 1,1, T, 1, 1, 1, FUNC>
      	prims(tid, nthreads, &myNghr, &myNghr, thisOutput, stepSize, channel, comm, ncclShmem->ptrs, 0);

      const int blockIdWithinALink = (bid % blocksPerLink);
      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        int realChunkSize = min(chunkSize, size-gridOffset);
        ALIGN_SIZE(realChunkSize, nthreads*sizeof(uint64_t)/sizeof(T));
        ssize_t chunkOffset = gridOffset + blockIdWithinALink * realChunkSize;// + bid*realChunkSize;
      	int nelem = min(realChunkSize, size-chunkOffset);

      	int rankDest = myRank;
      	ssize_t offset = chunkOffset + rankDest * size;

        if (thisInput + chunkOffset == thisOutput + offset) { // In place
          prims.directSend(thisInput+chunkOffset, offset, nelem);
          //prims.send(thisInput+chunkOffset, nelem);
        } else {
          prims.directCopySend(thisInput+chunkOffset, thisOutput+offset, offset, nelem);
          //prims.copySend(thisInput+chunkOffset, thisOutput+offset, nelem);
        }

      	rankDest = myNghr;
      	offset = chunkOffset + rankDest * size;
      	prims.directRecv(thisOutput+offset, offset, nelem);
      	//prims.recv(thisOutput+offset, nelem);
      }
#endif

    }
};

#else

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

