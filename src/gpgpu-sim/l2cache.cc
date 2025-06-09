// Copyright (c) 2009-2021, Tor M. Aamodt, Vijay Kandiah, Nikos Hardavellas,
// Mahmoud Khairy, Junrui Pan, Timothy G. Rogers
// The University of British Columbia, Northwestern University, Purdue
// University All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
// this
//    list of conditions and the following disclaimer;
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution;
// 3. Neither the names of The University of British Columbia, Northwestern
//    University nor the names of their contributors may be used to
//    endorse or promote products derived from this software without specific
//    prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <list>
#include <set>

#include "../abstract_hardware_model.h"
#include "../option_parser.h"
#include "../statwrapper.h"
#include "dram.h"
#include "gpu-cache.h"
#include "gpu-sim.h"
#include "histogram.h"
#include "l2cache.h"
#include "l2cache_trace.h"
#include "mem_fetch.h"
#include "mem_latency_stat.h"
#include "shader.h"

mem_fetch *partition_mf_allocator::alloc(new_addr_type addr,
                                         mem_access_type type, unsigned size,
                                         bool wr, unsigned long long cycle,
                                         unsigned long long streamID) const {
  assert(wr);
  mem_access_t access(type, addr, size, wr, m_memory_config->gpgpu_ctx);
  mem_fetch *mf = new mem_fetch(access, NULL, streamID, WRITE_PACKET_SIZE, -1,
                                -1, -1, m_memory_config, cycle);
  return mf;
}

mem_fetch *partition_mf_allocator::alloc(
    new_addr_type addr, mem_access_type type, const active_mask_t &active_mask,
    const mem_access_byte_mask_t &byte_mask,
    const mem_access_sector_mask_t &sector_mask, unsigned size, bool wr,
    unsigned long long cycle, unsigned wid, unsigned sid, unsigned tpc,
    mem_fetch *original_mf, unsigned long long streamID) const {
  mem_access_t access(type, addr, size, wr, active_mask, byte_mask, sector_mask,
                      m_memory_config->gpgpu_ctx);
  mem_fetch *mf = new mem_fetch(access, NULL, streamID,
                                wr ? WRITE_PACKET_SIZE : READ_PACKET_SIZE, wid,
                                sid, tpc, m_memory_config, cycle, original_mf);
  return mf;
}
memory_partition_unit::memory_partition_unit(unsigned partition_id,
                                             const memory_config *config,
                                             class memory_stats_t *stats,
                                             class gpgpu_sim *gpu)
    : m_id(partition_id),
      m_config(config),
      m_stats(stats),
      m_arbitration_metadata(config),
      m_gpu(gpu) {

  // pshyun {
  //m_dram = new dram_t(m_id, m_config, m_stats, this, gpu);
    m_dram = new ndc_t(m_id, m_config, m_stats, this, gpu);
    m_wrbk_mf_allocator = new partition_mf_allocator(config);
  // } pshyun

  m_sub_partition = new memory_sub_partition
      *[m_config->m_n_sub_partition_per_memory_channel];
  for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel;
       p++) {
    unsigned sub_partition_id =
        m_id * m_config->m_n_sub_partition_per_memory_channel + p;
    m_sub_partition[p] =
        new memory_sub_partition(sub_partition_id, m_config, stats, gpu);
  }
  
  // pshyun {
  unsigned int dram_cxl;
  unsigned int ndc_fill;
  sscanf(m_config->gpgpu_cxl_queue_config,"%u:%u", &dram_cxl,&ndc_fill);
  m_dram_cxl_queue = new fifo_pipeline<mem_fetch>("dram-to-cxl",0,dram_cxl);
  m_ndc_fill_queue = new fifo_pipeline<mem_fetch>("ndc_fill",0,ndc_fill);

  dram_total_cycle_cnt = 0;
  dram_acc_cnt = 0;
  dram_acc_from_l2_cnt = 0;
  dram_acc_from_cxl_cnt = 0;
  queue_full_cnt_returnq_cache_cycle = 0;
  queue_full_cnt_returnq_fill_cycle = 0;
  queue_full_cnt_returnq_ndc_cycle = 0;
  queue_full_cnt_ndc_fill_q = 0;
  queue_full_cnt_dram_cxl_q = 0;
  queue_full_cnt_mrqq = 0;
  queue_full_cnt_fill_mrqq = 0;
  ndc_hit_cnt = 0;
  ndc_miss_cnt = 0;
  ndc_rsv_fail_cnt = 0;
  ndc_evict_cnt = 0;
  ndc_data_port_busy_cnt = 0;
  ndc_fill_port_busy_cnt = 0;
  print_flag = 0;

  wrbk_borrow_credit = 0;
  global_acc_borrow_credit = 0;
  return_credit = 0;
  borrow_credit = 0;
  wrbk_return_credit = 0;
  global_acc_return_credit = 0;
  read_borrow_credit = std::vector<unsigned int>(24, 0);
  read_return_credit = std::vector<unsigned int>(24, 0);
  wrbk_access_cnt = 0;
  // } pshyun

}

void memory_partition_unit::handle_memcpy_to_gpu(
    size_t addr, unsigned global_subpart_id, mem_access_sector_mask_t mask) {
  unsigned p = global_sub_partition_id_to_local_id(global_subpart_id);
  std::string mystring = mask.to_string<char, std::string::traits_type,
                                        std::string::allocator_type>();
  MEMPART_DPRINTF(
      "Copy Engine Request Received For Address=%zx, local_subpart=%u, "
      "global_subpart=%u, sector_mask=%s \n",
      addr, p, global_subpart_id, mystring.c_str());
  m_sub_partition[p]->force_l2_tag_update(
      addr, m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle, mask);
}

memory_partition_unit::~memory_partition_unit() {
  delete m_dram;
  // pshyun {
    delete m_dram_cxl_queue;
    delete m_ndc_fill_queue;
  // } pshyun
  for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel;
       p++) {
    delete m_sub_partition[p];
  }
  delete[] m_sub_partition;
}

memory_partition_unit::arbitration_metadata::arbitration_metadata(
    const memory_config *config)
    : m_last_borrower(config->m_n_sub_partition_per_memory_channel - 1),
      m_private_credit(config->m_n_sub_partition_per_memory_channel, 0),
      m_shared_credit(0) {
  // each sub partition get at least 1 credit for forward progress
  // the rest is shared among with other partitions
  m_private_credit_limit = 1;
  m_shared_credit_limit = config->gpgpu_frfcfs_dram_sched_queue_size +
                          config->gpgpu_dram_return_queue_size -
                          (config->m_n_sub_partition_per_memory_channel - 1);
  if (config->seperate_write_queue_enabled)
    m_shared_credit_limit += config->gpgpu_frfcfs_dram_write_queue_size;
  if (config->gpgpu_frfcfs_dram_sched_queue_size == 0 or
      config->gpgpu_dram_return_queue_size == 0) {
    m_shared_credit_limit =
        0;  // no limit if either of the queue has no limit in size
  }
  assert(m_shared_credit_limit >= 0);
}

bool memory_partition_unit::arbitration_metadata::has_credits(
    int inner_sub_partition_id) const {
  int spid = inner_sub_partition_id;
  if (m_private_credit[spid] < m_private_credit_limit) {
    return true;
  } else if (m_shared_credit_limit == 0 ||
             m_shared_credit < m_shared_credit_limit) {
    return true;
  } else {
    return false;
  }
}

void memory_partition_unit::arbitration_metadata::borrow_credit(
    int inner_sub_partition_id) {
  int spid = inner_sub_partition_id;
  if (m_private_credit[spid] < m_private_credit_limit) {
    m_private_credit[spid] += 1;
  } else if (m_shared_credit_limit == 0 ||
             m_shared_credit < m_shared_credit_limit) {
    m_shared_credit += 1;
  } else {
    assert(0 && "DRAM arbitration error: Borrowing from depleted credit!");
  }
  m_last_borrower = spid;
}

void memory_partition_unit::arbitration_metadata::return_credit(
    int inner_sub_partition_id) {
  int spid = inner_sub_partition_id;
  if (m_private_credit[spid] > 0) {
    m_private_credit[spid] -= 1;
  } else {
    m_shared_credit -= 1;
  }
  assert((m_shared_credit >= 0) &&
         "DRAM arbitration error: Returning more than available credits!");
}

void memory_partition_unit::arbitration_metadata::print(FILE *fp) const {
  fprintf(fp, "private_credit = ");
  for (unsigned p = 0; p < m_private_credit.size(); p++) {
    fprintf(fp, "%d ", m_private_credit[p]);
  }
  fprintf(fp, "(limit = %d)\n", m_private_credit_limit);
  fprintf(fp, "shared_credit = %d (limit = %d)\n", m_shared_credit,
          m_shared_credit_limit);
}

bool memory_partition_unit::busy() const {
  bool busy = false;
  // pshyun {
  if (!m_request_tracker_dram.empty())
      busy = true;
  if (!m_request_tracker_ndc.empty())
      busy = true;
  // } pshyun
  for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel;
       p++) {
    if (m_sub_partition[p]->busy()) {
      busy = true;
    }
  }
  return busy;
}

void memory_partition_unit::cache_cycle(unsigned cycle) {
  for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel;
       p++) {
    m_sub_partition[p]->cache_cycle(cycle);
  }
}

void memory_partition_unit::visualizer_print(gzFile visualizer_file) const {
  m_dram->visualizer_print(visualizer_file);
  for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel;
       p++) {
    m_sub_partition[p]->visualizer_print(visualizer_file);
  }
}

// determine whether a given subpartition can issue to DRAM
bool memory_partition_unit::can_issue_to_dram(int inner_sub_partition_id) {
  int spid = inner_sub_partition_id;
  bool sub_partition_contention = m_sub_partition[spid]->dram_L2_queue_full();
  bool has_dram_resource = m_arbitration_metadata.has_credits(spid);

  MEMPART_DPRINTF(
      "sub partition %d sub_partition_contention=%c has_dram_resource=%c\n",
      spid, (sub_partition_contention) ? 'T' : 'F',
      (has_dram_resource) ? 'T' : 'F');

  return (has_dram_resource && !sub_partition_contention);
}

int memory_partition_unit::global_sub_partition_id_to_local_id(
    int global_sub_partition_id) const {
  return (global_sub_partition_id -
          m_id * m_config->m_n_sub_partition_per_memory_channel);
}

void memory_partition_unit::simple_dram_model_cycle() {
  // pop completed memory request from dram and push it to dram-to-L2 queue
  // of the original sub partition
  printf("[YH_DEBUG][ERROR] simple_dram_model_cycle is not allowed\n");
  if (!m_dram_latency_queue.empty() &&
      ((m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle) >=
       m_dram_latency_queue.front().ready_cycle)) {
    mem_fetch *mf_return = m_dram_latency_queue.front().req;
    if (mf_return->get_access_type() != L1_WRBK_ACC &&
        mf_return->get_access_type() != L2_WRBK_ACC) {
      mf_return->set_reply();

      unsigned dest_global_spid = mf_return->get_sub_partition_id();
      int dest_spid = global_sub_partition_id_to_local_id(dest_global_spid);
      assert(m_sub_partition[dest_spid]->get_id() == dest_global_spid);
      if (!m_sub_partition[dest_spid]->dram_L2_queue_full()) {
        if (mf_return->get_access_type() == L1_WRBK_ACC) {
          m_sub_partition[dest_spid]->set_done(mf_return);
          delete mf_return;
        } else {
          m_sub_partition[dest_spid]->dram_L2_queue_push(mf_return);
          mf_return->set_status(
              IN_PARTITION_DRAM_TO_L2_QUEUE,
              m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
          m_arbitration_metadata.return_credit(dest_spid);
          MEMPART_DPRINTF(
              "mem_fetch request %p return from dram to sub partition %d\n",
              mf_return, dest_spid);
        }
        m_dram_latency_queue.pop_front();
      }

    } else {
      this->set_done(mf_return);
      delete mf_return;
      m_dram_latency_queue.pop_front();
    }
  }

  // mem_fetch *mf = m_sub_partition[spid]->L2_dram_queue_top();
  // if( !m_dram->full(mf->is_write()) ) {
  // L2->DRAM queue to DRAM latency queue
  // Arbitrate among multiple L2 subpartitions
  int last_issued_partition = m_arbitration_metadata.last_borrower();
  for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel;
       p++) {
    int spid = (p + last_issued_partition + 1) %
               m_config->m_n_sub_partition_per_memory_channel;
    if (!m_sub_partition[spid]->L2_dram_queue_empty() &&
        can_issue_to_dram(spid)) {
      mem_fetch *mf = m_sub_partition[spid]->L2_dram_queue_top();
      if (m_dram->full(mf->is_write())) break;

      m_sub_partition[spid]->L2_dram_queue_pop();
      MEMPART_DPRINTF(
          "Issue mem_fetch request %p from sub partition %d to dram\n", mf,
          spid);
      dram_delay_t d;
      d.req = mf;
      d.ready_cycle = m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle +
                      m_config->dram_latency;
      m_dram_latency_queue.push_back(d);
      mf->set_status(IN_PARTITION_DRAM_LATENCY_QUEUE,
                     m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
      m_arbitration_metadata.borrow_credit(spid);
      break;  // the DRAM should only accept one request per cycle
    }
  }
  //}
}

void memory_partition_unit::dram_cycle() {
  // pop completed memory request from dram and push it to dram-to-L2 queue
  // of the original sub partition

  // pshyun {
  // --------------------------------------------------
  // 1) for m_cxl_latency_queue
  // --------------------------------------------------
  if( !m_cxl_latency_queue.empty() && ( (m_gpu->gpu_sim_cycle+m_gpu->gpu_tot_sim_cycle) >= m_cxl_latency_queue.front().ready_cycle )) {
    //check type and path
    mem_fetch* mf = m_cxl_latency_queue.front().req;
    //printf("[KSH_DEBUG][cxl_latency_queue->pop] uid: %d, ret = %d, req = %d\n", mf->get_request_uid(), mf->get_cxl_req_type(), mf->get_cxl_ret_path());

    unsigned dest_global_spid = mf->get_sub_partition_id(); 
    int dest_spid = global_sub_partition_id_to_local_id(dest_global_spid); 
    assert(m_sub_partition[dest_spid]->get_id() == dest_global_spid); 

    if (mf->get_cxl_ret_path() == CXL_NONE) {
      // CXL WB case
      m_cxl_latency_queue.pop_front();    //removed
      
      if(mf->get_access_type() != NDC_WRBK_ACC) {
          printf("[YH_DEBUG][mem_partition[%d]][mem_sub_partition[%d]] wrong access type (or cxl_ret_path) was detected. uid: %d\n", m_id, dest_spid, mf->get_request_uid());
          mf->print(stdout);
          assert(0);
      }
      //for_removing_tracker_ndc:delete(mf); //FIXME
      delete_new_mf(mf);
    }
    else if (mf->get_cxl_ret_path() == CXL_DRAM) {
      // prefetch / prediction or L2_DRAM and L2 already sent
      if (!ndc_fill_queue_full()) {
        if(mf->get_access_type() == NDC_WR_ALLOC_R) {
            // cxl read -> dram fill(write)
            mf->set_type_to_write();
        }
        //YH_DEBUG:printf("[YH_DEBUG][dram_cycle][mem_partition[%d]][mem_sub_partition[%d]] ret_path = DRAM(predict)\n", m_id, dest_spid);
        mf->set_status(IN_PARTITION_CXL_TO_DRAM_QUEUE, m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
        // TODO:if(mf->get_request_uid() == TGT_UID)
        // TODO:    printf("[YH_DEBUG] Moved info. uid: %d, status: %s \n", mf->get_request_uid(), Status_str[mf->m_status]);

        ndc_fill_queue_push(mf);
        // yhyang: 250221_16_for_hack: //  cxl read -> dram fill(write)
        // yhyang: 250221_16_for_hack: dram_req_t *dr = new dram_req_t(mf);
        // yhyang: 250221_16_for_hack: bool fill_success = m_dram->ndc_fill_access(dr);
        // yhyang: 250221_16_for_hack: if (fill_success) delete dr;

        m_cxl_latency_queue.pop_front();
      }
      else {
        queue_full_cnt_ndc_fill_q++;
      }
    }
    else if (mf->get_cxl_ret_path() == CXL_L2) {
      // NDC hit or Linefill (DRAM request is already sent) case
      if (!m_sub_partition[dest_spid]->dram_L2_queue_full()) {
        //YH_DEBUG:printf("[YH_DEBUG][dram_cycle][mem_partition[%d]][mem_sub_partition[%d]] ret_path = L2(NDC hit or linefill)\n", m_id, dest_spid);
        mem_fetch *mf_orig = find_orig_mf(mf);
        mf->set_status(IN_PARTITION_DRAM_TO_L2_QUEUE, m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
        mf->set_l2_done(true);
        delete_new_mf(mf);
        m_sub_partition[dest_spid]->dram_L2_queue_push(mf_orig);
        m_cxl_latency_queue.pop_front();
      }
    }
    else if (mf->get_cxl_ret_path() == CXL_L2_DRAM) {
      if ((!ndc_fill_queue_full()) || (!m_sub_partition[dest_spid]->dram_L2_queue_full())) {
        mem_fetch *mf_orig = find_orig_mf(mf);

        //cxl read -> dram fill(write)
        mf->set_type_to_write();
        mf->set_status(IN_PARTITION_CXL_TO_DRAM_QUEUE, m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
        mf->set_cxl_req_type(CXL_RD_LINE_FILL);
        //yhyang:why?:mf->set_ndc_resp(NDC_INVALID);
        if ((!ndc_fill_queue_full()) && (!m_sub_partition[dest_spid]->dram_L2_queue_full())) {
            //printf("[PSH_DEBUG][%d][cxl_latency_queue->pop : go to l2 and ndc_fill_queue] uid: %d, mf_orig uid %d\n", m_gpu->gpu_sim_cycle, mf->get_request_uid(), mf_orig->get_request_uid());
            ndc_fill_queue_push(mf);
            // yhyang: 250221_16_for_hack: dram_req_t *dr = new dram_req_t(mf);
            // yhyang: 250221_16_for_hack: bool fill_success = m_dram->ndc_fill_access(dr);
            // yhyang: 250221_16_for_hack: if (fill_success) delete dr;

            mf->set_l2_done(true);
            //dram_access_is_required:delete_new_mf(mf);
            m_sub_partition[dest_spid]->dram_L2_queue_push(mf_orig);

            //delete_new_mf(mf); // PLEASE FIXME

            mf->set_cxl_ret_path(CXL_NONE);
            m_cxl_latency_queue.pop_front();
        }
        //check if NDC_FILL_DONE treatment is wrong due to l2_done flag
        else if (!ndc_fill_queue_full()) {

            ndc_fill_queue_push(mf);
            // yhyang: 250221_16_for_hack: dram_req_t *dr = new dram_req_t(mf);
            // yhyang: 250221_16_for_hack: bool fill_success = m_dram->ndc_fill_access(dr);
            // yhyang: 250221_16_for_hack: if (fill_success) delete dr;

            mf->set_cxl_ret_path(CXL_L2);
        }
        else if (!m_sub_partition[dest_spid]->dram_L2_queue_full()) {
            queue_full_cnt_ndc_fill_q++;
            mf->set_l2_done(true);
            //dram_access_is_required:delete_new_mf(mf);
            m_sub_partition[dest_spid]->dram_L2_queue_push(mf_orig);
            delete_new_mf(mf);
            mf->set_cxl_ret_path(CXL_DRAM);
        }
        else {
            queue_full_cnt_ndc_fill_q++;
            // Invalid
        }
#ifdef YH_DEBUG
        printf("[YH_DEBUG] CXL_L2_DRAM queue mf information\n");
        if(m_id == 0) mf->print(stdout);
#endif // YH_DEBUG
      }
    }
  }
  // } pshyun

  // pshyun {
  // --------------------------------------------------
  // 2) for pushing to CXL latency queue
  //    DRAM->CXL path. dram_cxl_queue -> cxl_latency_queue
  // --------------------------------------------------
  if (!dram_cxl_queue_empty()) {
    mem_fetch *mf = dram_cxl_queue_top();
    dram_cxl_queue_pop();
    MEMPART_DPRINTF("Issue mem_fetch request %p from dram to cxl queue\n", mf);
    dram_delay_t d;
    d.req = mf;
    d.ready_cycle = m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle + m_config->cxl_latency;
    m_cxl_latency_queue.push_back(d);
    // printf("[PSH_DEBUG][%d][dram_cxl_queue->cxl_latency_queue] uid: %d, ret = %d, req = %d, now_cycle = %d, ready_cycle = %d\n", 
    //   m_gpu->gpu_sim_cycle, mf->get_request_uid(), mf->get_cxl_req_type(), mf->get_cxl_ret_path(),m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle, d.ready_cycle);
    mf->set_status(IN_PARTITION_CXL_LATENCY_QUEUE, m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
  }
  // } pshyun

  // pshyun {
  // --------------------------------------------------
  // 3) for popping returnq
  //    3-1) returnq -> delete (fill_done)
  //    3-2) returnq -> dram_L2_queue
  //    3-3) returnq -> dram_cxl_queue
  // --------------------------------------------------
  mem_fetch *mf_return = m_dram->return_queue_top();
  if (mf_return) {
    unsigned dest_global_spid = mf_return->get_sub_partition_id();
    int dest_spid = global_sub_partition_id_to_local_id(dest_global_spid);
    assert(m_sub_partition[dest_spid]->get_id() == dest_global_spid);
    if (m_config->m_L3_NDC_config.disabled()) { // original
      if (!m_sub_partition[dest_spid]->dram_L2_queue_full()) {
        if (mf_return->get_access_type() == L1_WRBK_ACC) {
          m_sub_partition[dest_spid]->set_done(mf_return);
          delete mf_return;
        } else {
          m_sub_partition[dest_spid]->dram_L2_queue_push(mf_return);
          mf_return->set_status(IN_PARTITION_DRAM_TO_L2_QUEUE,
                                m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
          m_arbitration_metadata.return_credit(dest_spid);
          MEMPART_DPRINTF(
              "mem_fetch request %p return from dram to sub partition %d\n",
              mf_return, dest_spid);
        }
        m_dram->return_queue_pop();
      }
    } else { // l3_ndc_cache enable
      // ---------------------------------------------------------------------------------------------------------------
      // NDC case
      // ---------------------------------------------------------------------------------------------------------------
      // WR or RD line fill. for identifying 1) and 2), check l2_done flag...?
      // --------------------------------------------------------------------------------------------------------------
      // 1) Read NDC_MISS -> CXL READ -> NDC_MISS + NDC_RD_LINE_FILL done case
      //      : [1] NDC_MISS        : (1) send to CXL, (2) set l2_done after CXL latency_queue, (3) clean orig request after CXL latency queue
      //      : [2] NDC_RD_LINE_FILL: (1) return credit, (2) set dram_done, (6) clean this resp
      // 2) Read NDC_MISS -> Wait at MSHR -> NDC_RD_LINE_FILL done case
      //      : [1] NDC_RD_LINE_FILL: (1) set l2_done, (2) set dram_done, (3) return credit, (4) clean orig request, (5) send to L2, (6) clean this resp
      // 3) Write NDC_MISS -> CXL READ(CXL_WR_LINE_FILL/NDC_WR_ALLOC_R) -> orig(CXL_NONE / NDC_INVALID) + new(NDC_FILL_DONE) response done case
      //      : [1-1] NDC_INVALID (L1/L2_WRBK_ACC_W)  :  (1) clear orig request ,  (2) set l2_done
      //      : [1-2] NDC_INVALID (GLOBAL_ACC_W)      :  (1) set l2_done,  (2) clean orig request, (3) send to L2
      //      : [2] NDC_FILL_DONE  :  (1) return credit, (2) set dram_done, (3) just delete
      // 4) Write NDC_MISS -> Wait at MSHR -> NDC_INVALID + NDC_FILL_DONE response done case
      //      : [1-1] NDC_INVALID (L1/L2_WRBK_ACC_W)  :  (1) clear orig request ,  (2) set l2_done
      //      : [1-2] NDC_INVALID (GLOBAL_ACC_W)      :  (1) set l2_done,  (2) clean orig request, (3) send to L2
      //      : [2] NDC_FILL_DONE  :  (1) return credit, (2) set dram_done, (3) just delete
      if ((mf_return->get_ndc_resp() == NDC_FILL_DONE)) {
          if((!mf_return->get_l2_done()) && (mf_return->get_cxl_req_type() == CXL_RD_LINE_FILL)) {
              // case 2-1
              // 2) Read NDC_MISS -> Wait at MSHR -> NDC_RD_LINE_FILL done case
              //      : [1] NDC_RD_LINE_FILL: (1) set l2_done, (2) set dram_done, (3) return credit, (4) clean orig request, (5) send to L2, (6) clean this resp
              if (!m_sub_partition[dest_spid]->dram_L2_queue_full())
              {
                  // send mf_orig to L2
                  mem_fetch *mf_orig = find_orig_mf(mf_return);
                  mf_orig->set_status(IN_PARTITION_DRAM_TO_L2_QUEUE, m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
                  m_sub_partition[dest_spid]->dram_L2_queue_push(mf_orig);
                  // delete mf_new
                  mf_return->set_l2_done(true);
                  mf_return->set_dram_done(true);
                  m_arbitration_metadata.return_credit(dest_spid);
                  return_credit++;
                  delete_new_mf(mf_return);
                  m_dram->return_queue_pop();
              }
          } else {
              // case 1-2, 3-2, 4-2
              // fill done. L2 is already done
              // delete mf_new
              // distinguish sector_request or not
              
              mf_return->set_dram_done(true);
              m_arbitration_metadata.return_credit(dest_spid);
              return_credit++;
              // if((mf_return->get_access_type()==NDC_WR_ALLOC_R)) {
              //   printf("[PSH_DEBUG][%d][m_id%d]Find GLOBAL_ACC_W miss and mshr-hit fill-done\n",m_gpu->gpu_sim_cycle, m_id);
              //   mf_return->print(stdout);
              //   global_acc_return_credit++;
              // } else if((mf_return->get_access_type()==NDC_LINEFILL_W) && (mf_return->get_cxl_ret_path() == CXL_DRAM)){
              //   printf("[PSH_DEBUG][%d][m_id%d]Find GLOBAL_ACC_W miss and mshr-miss fill-done\n",m_gpu->gpu_sim_cycle, m_id);
              //   mf_return->print(stdout);
              //   global_acc_return_credit++;
              // }
              delete_new_mf(mf_return);   // just deleted
              m_dram->return_queue_pop();
          }
      }
      // dram_L2_queue_full -> hit&dram_L2_queue_full / miss&dram_cxl_queue_full
      else if ((mf_return->get_ndc_resp() == NDC_HIT)) {

          mem_fetch *mf_orig = find_orig_mf(mf_return);
          //L1/L2 WRBK HIT
          if (mf_return->get_access_type() == L1_WRBK_ACC || mf_return->get_access_type() == L2_WRBK_ACC) {
              mf_return->set_l2_done(true);
              mf_return->set_dram_done(true);
              m_sub_partition[dest_spid]->set_done(mf_return->get_original_mf());
              m_arbitration_metadata.return_credit(dest_spid);
              return_credit++;
              delete mf_orig;                                   // TODO need to check
              delete_new_mf(mf_return);
              m_dram->return_queue_pop();
          } else if (!m_sub_partition[dest_spid]->dram_L2_queue_full()) {
              // send mf_orig to L2
              mf_orig->set_status(IN_PARTITION_DRAM_TO_L2_QUEUE, m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
              mf_orig->set_reply(); // FIXME
              m_sub_partition[dest_spid]->dram_L2_queue_push(mf_orig);
              MEMPART_DPRINTF("mem_fetch request %p return from dram to sub partition %d\n", mf_return, dest_spid);
              // delete mf_new
              mf_return->set_l2_done(true);
              mf_return->set_dram_done(true);
              delete_new_mf(mf_return);
              m_dram->return_queue_pop();
              m_arbitration_metadata.return_credit(dest_spid);  // return credit
              return_credit++;

              //debug
              if(mf_return->get_access_type()==GLOBAL_ACC_W) global_acc_return_credit++;
          }
      } else if (mf_return->get_ndc_resp() == NDC_MISS) {

          // case1-1
          // NDC miss read.
          // NDC_WR_ALLOC_R has CXL_NONE

          if (!(mf_return->get_cxl_ret_path() == CXL_NONE)) {
              if (!dram_cxl_queue_full()) {
                  assert (mf_return->get_cxl_ret_path() != CXL_DRAM); // CXL_DRAM is only for write. WR_MISS should not be controlled here 
                  dram_cxl_queue_push(mf_return);
                  mf_return->set_status(IN_PARTITION_DRAM_TO_CXL_QUEUE, m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
                  // not_here:m_arbitration_metadata.return_credit(dest_spid);
                  MEMPART_DPRINTF("mem_fetch request %p return from dram to sub partition %d for CXL access\n", mf_return, dest_spid);
                  m_dram->return_queue_pop();
              }
              else {
                  queue_full_cnt_dram_cxl_q++;
              }
          } else {  // NDC miss write ack
              // write request to DRAM created by mf_adaptor. It is required to respond to L2 if GLOBAL_ACC_W. If L1/L2 WRBK case, no ack.
              // mf_new maded by mf_adaptor should be deleted here
              assert(mf_return->get_access_type() != NDC_WR_ALLOC_R);   //NDC_WR_ALLOC_R type is created by NDC
              mem_fetch *mf_orig = find_orig_mf(mf_return);

              if (!((mf_return->get_access_type() == L1_WRBK_ACC) || (mf_return->get_access_type() == L2_WRBK_ACC))) {
                  // case 3-1-2, case4-1-2
                  // GLOBAL_ACC_W case. Need to response to L2
                  if (!m_sub_partition[dest_spid]->dram_L2_queue_full()) {
                      if ((mf_return->get_request_uid() == TGT_UID) || (mf_orig->get_request_uid() == TGT_UID)) {
                          printf("[YH_DEBUG][%d][NDC_MISS][uid:%d]\n", m_gpu->gpu_sim_cycle, mf_return->get_request_uid());
                          mf_return->print(stdout);
                          mf_orig->print(stdout);
                      }
                      mf_return->set_l2_done(true);
                      mf_return->set_dram_done(true);
                      // moved_to_dram_L2_queue_pop_step:mf->set_reply();    // set_reply for GLOBAL_ACC_W
                      mf_orig->set_status(IN_PARTITION_DRAM_TO_L2_QUEUE, m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
                      mf_orig->set_reply(); // pshyun : FIXME
                      delete_new_mf(mf_return);  // this will deleted here even though fill request is not completed. so set dram_done = true
                                                  // credit will be returned by fill_done response
                      m_sub_partition[dest_spid]->dram_L2_queue_push(mf_orig);
                      m_dram->return_queue_pop();
                  }
              } else {
                  // case 3-1-1, case4-1-1 (L1_WRBK, L2_WRBK)
                  
                  mf_return->set_l2_done(true);
                  mf_return->set_dram_done(true);                    // set dram_done also for removing WRBK request's info. There is no mapping info between dram_req and NDC miss->fill request
                  m_sub_partition[dest_spid]->set_done(mf_orig);  // original WB request is deleted. no credit return. WR_LINE_FILL_DONE will return credit
                  delete_new_mf(mf_return);   // mf_new deleted at here
                  m_dram->return_queue_pop();
              }
          }
      } else if (mf_return->get_access_type() == NDC_WR_ALLOC_R) {
          if (!dram_cxl_queue_full()) {
              // created by NDC during write miss
              mf_return->set_l2_done(true);

              //mf_return->set_ndc_resp(NDC_MISS);
              mf_return->set_cxl_req_type(CXL_WR_LINE_FILL);
              mf_return->set_cxl_ret_path(CXL_DRAM);

              //printf("[PSH_DEBUG][%d][mid%d][NDC_WR_ALLOC_R] : ret_path = %d\n", m_gpu->gpu_sim_cycle, m_id, mf_return->get_cxl_ret_path());
              //mf_return->print(stdout);

              dram_cxl_queue_push(mf_return);
              mf_return->set_status(IN_PARTITION_DRAM_TO_CXL_QUEUE, m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
              //not_here:m_arbitration_metadata.return_credit(dest_spid);
              MEMPART_DPRINTF("mem_fetch request %p return from dram to sub partition %d for CXL access\n", mf_return, dest_spid);
              m_request_tracker_ndc.insert(mf_return);    // insert to tracker_ndc
              m_dram->return_queue_pop();
          }
          else {
              queue_full_cnt_dram_cxl_q++;
          }
      } else if (mf_return->get_access_type() == NDC_WRBK_ACC) {
          if (!dram_cxl_queue_full()) {
#ifdef YH_DEBUG
                    printf("[YH_DEBUG][%d][NDC_WRBK_ACC] NDC_WRBK_ACC is detected uid: %d\n", m_gpu->gpu_sim_cycle, mf_return->get_request_uid());
#endif // YH_DEBUG
              // does not need to respond to L2 or re-write to DRAM. only need to write to CXL
              mf_return->set_l2_done(true);
              mf_return->set_dram_done(true);
              mf_return->set_cxl_req_type(CXL_WB);

              dram_cxl_queue_push(mf_return);
              mf_return->set_status(IN_PARTITION_DRAM_TO_CXL_QUEUE, m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
              // not_here:m_arbitration_metadata.return_credit(dest_spid);
              MEMPART_DPRINTF("mem_fetch request %p return from dram to sub partition %d for CXL WB access\n", mf_return, dest_spid);
              m_request_tracker_ndc.insert(mf_return);
              m_dram->return_queue_pop();
              ndc_evict_cnt += 1;
          }
          // yhyang:} else if ((mf_return->get_ndc_resp() == NDC_EVICT)) {
          // yhyang:    // yhyang invalid condition
          // yhyang:    assert(0);
          else {
              queue_full_cnt_dram_cxl_q++;
          }
      } else {
          printf("[YH_DEBUG][%d][NDC_INVALID] UNDEFINED response from NDC.uid: %d\n", m_gpu->gpu_sim_cycle, mf_return->get_request_uid());
          mf_return->print(stdout);
          assert(0);
      }
    }
  } else {
    m_dram->return_queue_pop();
  }

  // --------------------------------------------------
  // 4) for dram cycle
  // --------------------------------------------------
  m_dram->cycle();
  m_dram->dram_log(SAMPLELOG);

  // pshyun {
  // --------------------------------------------------
  // 5) for L2_dram_queue -> dram_latency_queue
  // 6) for ndc_fill_queue -> dram_latency_queue_from_cxl
  // --------------------------------------------------

  // mem_fetch *mf = m_sub_partition[spid]->L2_dram_queue_top();
  // if( !m_dram->full(mf->is_write()) ) {
  // L2->DRAM queue to DRAM latency queue
  // Arbitrate among multiple L2 subpartitions
  if(m_config->ndc_bypass) {
    assert(0); // no_ndc_bypass
    int last_issued_partition = m_arbitration_metadata.last_borrower();
    for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel; p++) {
      int spid = (p + last_issued_partition + 1) % m_config->m_n_sub_partition_per_memory_channel;
      if (!m_sub_partition[spid]->L2_dram_queue_empty() && can_issue_to_dram(spid)) {
        if (!m_config->m_L3_NDC_config.disabled()) {
          mem_fetch *mf_orig = m_sub_partition[spid]->L2_dram_queue_top();
          mem_fetch *mf_new = create_new_mf(mf_orig);
          //if (m_dram->full(mf->is_write())) break; // we don't need to send request to dram, only cxl

          m_sub_partition[spid]->L2_dram_queue_pop();
          MEMPART_DPRINTF("Issue mem_fetch request %p from sub partition %d to dram\n", mf_new, spid);
          mf_new->set_cxl_ret_path(CXL_L2);
          mf_new->set_dram_done(true);
          dram_delay_t d;
          d.req = mf_new;
          d.ready_cycle = m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle +
                          m_config->cxl_latency;
          m_cxl_latency_queue.push_back(d);
          mf_new->set_status(IN_PARTITION_DRAM_LATENCY_QUEUE,
                        m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
          //m_arbitration_metadata.borrow_credit(spid); // we don't need to send request to dram, only cxl
          break;  // the DRAM should only accept one request per cycle
        } else {
          mem_fetch *mf = m_sub_partition[spid]->L2_dram_queue_top();
          if (m_dram->full(mf->is_write())) break;

          m_sub_partition[spid]->L2_dram_queue_pop();
          MEMPART_DPRINTF("Issue mem_fetch request %p from sub partition %d to dram\n", mf, spid);
          dram_delay_t d;
          d.req = mf;
          d.ready_cycle = m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle +
                          m_config->dram_latency;
          m_dram_latency_queue.push_back(d);
          mf->set_status(IN_PARTITION_DRAM_LATENCY_QUEUE,
                        m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
          m_arbitration_metadata.borrow_credit(spid);
          break;  // the DRAM should only accept one request per cycle
        }
      } else if(!m_sub_partition[spid]->L2_dram_queue_empty()){
        //printf("[PSH_DEBUG]Can't issue to dram, no-credit\n");
      }
    }
  // } else if (!m_dram->full()) {  // pshyun : dram full checking is under the condition  
  } else {
    int last_issued_partition = m_arbitration_metadata.last_borrower();
    for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel; p++) {
      int spid = (p + last_issued_partition + 1) % m_config->m_n_sub_partition_per_memory_channel; 
      // 1) L2->DRAM
      // create new mf after l2_dram_queue for independent operation
      if (!m_sub_partition[spid]->L2_dram_queue_empty() && can_issue_to_dram(spid)) {
        if (!m_config->m_L3_NDC_config.disabled()) {
            mem_fetch *mf_orig = m_sub_partition[spid]->L2_dram_queue_top();
            if (!m_sub_partition[spid]->m_wrbk_breakdown_queue.empty()){
              if (m_dram->full(mf_orig->is_write())) break;
              mem_fetch *mf_new = m_sub_partition[spid]->m_wrbk_breakdown_queue.front();
              dram_delay_t d;
              d.req = mf_new;
              d.ready_cycle = m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle + m_config->dram_latency;
              m_dram_latency_queue.push_back(d);
              mf_new->set_status(IN_PARTITION_DRAM_LATENCY_QUEUE, m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
              m_arbitration_metadata.borrow_credit(spid);
              borrow_credit++;
              m_sub_partition[spid]->m_wrbk_breakdown_queue.pop_front();
              break;
            }
            else if(mf_orig->get_access_type() == L1_WRBK_ACC || mf_orig->get_access_type() == L2_WRBK_ACC) {
              if (m_dram->full(mf_orig->is_write())) break;

              std::vector<mem_fetch *> mf_news;
              mf_news = breakdown_wrbk_request_to_sector_requests(mf_orig);
              for(int i = 0; i < mf_news.size(); i++) {
                if(i == 0) {
                  mem_fetch *mf_new = mf_news[i];
                  dram_delay_t d;
                  d.req = mf_new;
                  d.ready_cycle = m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle + m_config->dram_latency;
                  m_dram_latency_queue.push_back(d);
                  mf_new->set_status(IN_PARTITION_DRAM_LATENCY_QUEUE, m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
                  m_arbitration_metadata.borrow_credit(spid);
                  borrow_credit++;
                } else {
                  m_sub_partition[spid]->m_wrbk_breakdown_queue.push_back(mf_news[i]);
                }
              }

              m_sub_partition[spid]->L2_dram_queue_pop();
              break;
            } else {
              if (m_dram->full(mf_orig->is_write())) break; // pshyun : GLOBAL_W or GLOBAL_R or L2_WR_ALLOC_R
              //wrong...opposite_way: if(mf_new->get_data_size() < 64) {
              //wrong...opposite_way:     // set minimum access granularity to acces NDC is 64B...
              //wrong...opposite_way:     mf_new->set_data_size(64);
              //wrong...opposite_way: }
              m_sub_partition[spid]->L2_dram_queue_pop();
              mem_fetch *mf_new = create_new_mf(mf_orig);

              MEMPART_DPRINTF("Issue mem_fetch request %p from sub partition %d to dram\n", mf_new, spid);
              dram_delay_t d;
              // d.req = mf;
              d.req = mf_new;
              d.ready_cycle = m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle + m_config->dram_latency;
              m_dram_latency_queue.push_back(d);
              mf_new->set_status(IN_PARTITION_DRAM_LATENCY_QUEUE, m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
              m_arbitration_metadata.borrow_credit(spid);
              borrow_credit++;

              //debug
              // if(mf_new->get_access_type() == GLOBAL_ACC_W) {
              //   printf("[PSH_DEBUG][%d][m_id%d] First L2 to Dram GLOBAL_ACC_W request\n",m_gpu->gpu_sim_cycle, m_id);
              //   mf_new->print(stdout);
              //   global_acc_borrow_credit++;
              // } 

              // required for predcitor / prefetcher
              break;  // the DRAM should only accept one request per cycle
            }
        } else {
            mem_fetch *mf = m_sub_partition[spid]->L2_dram_queue_top();
            if (m_dram->full(mf->is_write())) break; // pshyun : add condition

            m_sub_partition[spid]->L2_dram_queue_pop();
            MEMPART_DPRINTF("Issue mem_fetch request %p from sub partition %d to dram\n", mf, spid);
            dram_delay_t d;
            d.req = mf;
            d.ready_cycle = m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle + m_config->dram_latency;
            m_dram_latency_queue.push_back(d);
            mf->set_status(IN_PARTITION_DRAM_LATENCY_QUEUE, m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
            m_arbitration_metadata.borrow_credit(spid);
            borrow_credit++;
            break;  // the DRAM should only accept one request per cycle
        }
      } else if(!m_sub_partition[spid]->L2_dram_queue_empty()) {
        //printf("[PSH_DEBUG]Can't issue to dram, No-Credit\n");
      }
    }
    // 2) CXL->DRAM path fill or prefetch/predict (not related to sub_partition anymore)
    if (!ndc_fill_queue_empty()) {
      mem_fetch *mf = ndc_fill_queue_top();
      ndc_fill_queue_pop();

      MEMPART_DPRINTF("Issue mem_fetch request %p ndc_fill\n", mf);
      dram_delay_t d;
      d.req = mf;
      d.ready_cycle = m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle + m_config->dram_latency;
      m_dram_latency_queue_from_cxl.push_back(d);
      mf->set_status(IN_PARTITION_DRAM_LATENCY_QUEUE, m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
      //borrow credit is not required because this request was created by NDC and first request was already borrowed credit.

      // required for predcitor / prefetcher
    }
  }
  // } pshyun

  // pshyun {
  // --------------------------------------------------
  // 7) dram_latency_queue -> m_dram->m_dram->mrqq
  // 8) dram_latency_queue_from_cxl -> m_dram->fill_mrqq
  // --------------------------------------------------
  if (m_dram->full_from_cxl()) {
    queue_full_cnt_fill_mrqq++;
  }
  if (m_dram->full(m_dram_latency_queue.front().req->is_write())) {
    queue_full_cnt_mrqq++;
  }

  
  // DRAM latency queue
  int cond1 = !m_dram_latency_queue.empty() && ( (m_gpu->gpu_sim_cycle+m_gpu->gpu_tot_sim_cycle) >= m_dram_latency_queue.front().ready_cycle ) && !m_dram->full(m_dram_latency_queue_from_cxl.front().req->is_write());
  int cond2 = !m_dram_latency_queue_from_cxl.empty() && ((m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle) >= m_dram_latency_queue_from_cxl.front().ready_cycle) && !m_dram->full_from_cxl();
  if ((cond1) || (cond2)) {
    if (cond2) {
      mem_fetch *mf = m_dram_latency_queue_from_cxl.front().req;
      m_dram_latency_queue_from_cxl.pop_front();
      m_dram->push_from_cxl(mf);
      dram_acc_cnt++;
      dram_acc_from_cxl_cnt++;
    } else if(cond1) {
      mem_fetch *mf = m_dram_latency_queue.front().req;
      m_dram_latency_queue.pop_front();
      m_dram->push(mf);
      dram_acc_cnt++;
      dram_acc_from_l2_cnt++;
    }
  }
  
  // if(m_gpu->gpu_sim_cycle % 50000 == 0) {
  //   printf("[PSH_DEBUG][%d][m_id%d] Borrow_Credit : %d(%d), Return_Credit : %d(%d)\n",m_gpu->gpu_sim_cycle, m_id, borrow_credit, global_acc_borrow_credit, return_credit, global_acc_return_credit);
  // }

  // if (!m_dram_latency_queue_from_cxl.empty() && (!m_dram->full_from_cxl())) {
  //   printf("[PSH_DEBUG][m_dram_latency_queue_from_cxl compare] uid : %d, cycle : %d, ready_cycle : %d\n", m_dram_latency_queue_from_cxl.front().req->get_request_uid(),
  //   (m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle)
  //    , m_dram_latency_queue_from_cxl.front().ready_cycle);
  // }
  //tracker
  // if(m_gpu->gpu_sim_cycle > 999) {
  //   if((m_gpu->gpu_sim_cycle % 5000) == 0) {
  //     if((print_flag == 0)) {
  //       printf("Memory Sub Parition %u: pending memory requests:\n", m_id);
  //       if (!m_request_tracker_dram.empty()) {
  //         for (std::set<mem_fetch *>::const_iterator r = m_request_tracker_dram.begin();
  //             r != m_request_tracker_dram.end(); ++r) {
  //           mem_fetch *mf = *r;
  //           if (mf)
  //             mf->print(stdout);
  //           else
  //             printf(" <NULL mem_fetch?>\n");
  //         }
  //       } else {
  //         printf("Memory Sub Parition %u: none-pending memory requests:\n", m_id);
  //       }
  //       print_flag = 1;
  //     } 
  //   } else if(((m_gpu->gpu_sim_cycle % 5000) == 1)) {
  //     if((print_flag == 0)) {
  //       printf("Memory Sub Parition %u: pending memory requests:\n", m_id);
  //       if (!m_request_tracker_dram.empty()) {
  //         for (std::set<mem_fetch *>::const_iterator r = m_request_tracker_dram.begin();
  //             r != m_request_tracker_dram.end(); ++r) {
  //           mem_fetch *mf = *r;
  //           if (mf)
  //             mf->print(stdout);
  //           else
  //             printf(" <NULL mem_fetch?>\n");
  //         }
  //       } else {
  //         printf("Memory Sub Parition %u: none-pending memory requests:\n", m_id);
  //       }
  //     } else {
  //       print_flag = 0;
  //     }
  //   }
  // // } pshyun 
  // }
}

void memory_partition_unit::set_done(mem_fetch *mf) {
  unsigned global_spid = mf->get_sub_partition_id();
  int spid = global_sub_partition_id_to_local_id(global_spid);
  assert(m_sub_partition[spid]->get_id() == global_spid);
  if (mf->get_access_type() == L1_WRBK_ACC ||
      mf->get_access_type() == L2_WRBK_ACC) {
    m_arbitration_metadata.return_credit(spid);
    return_credit++;
    MEMPART_DPRINTF(
        "mem_fetch request %p return from dram to sub partition %d\n", mf,
        spid);
  }
  m_sub_partition[spid]->set_done(mf);
}

void memory_partition_unit::set_dram_power_stats(
    unsigned &n_cmd, unsigned &n_activity, unsigned &n_nop, unsigned &n_act,
    unsigned &n_pre, unsigned &n_rd, unsigned &n_wr, unsigned &n_wr_WB,
    unsigned &n_req) const {
  m_dram->set_dram_power_stats(n_cmd, n_activity, n_nop, n_act, n_pre, n_rd,
                               n_wr, n_wr_WB, n_req);
}

void memory_partition_unit::print(FILE *fp) const {
  fprintf(fp, "Memory Partition %u: \n", m_id);
  for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel;
       p++) {
    m_sub_partition[p]->print(fp);
  }
  fprintf(fp, "In Dram Latency Queue (total = %zd): \n",
          m_dram_latency_queue.size());
  for (std::list<dram_delay_t>::const_iterator mf_dlq =
           m_dram_latency_queue.begin();
       mf_dlq != m_dram_latency_queue.end(); ++mf_dlq) {
    mem_fetch *mf = mf_dlq->req;
    fprintf(fp, "Ready @ %llu - ", mf_dlq->ready_cycle);
    if (mf)
      mf->print(fp);
    else
      fprintf(fp, " <NULL mem_fetch?>\n");
  }
  m_dram->print(fp);
}

memory_sub_partition::memory_sub_partition(unsigned sub_partition_id,
                                           const memory_config *config,
                                           class memory_stats_t *stats,
                                           class gpgpu_sim *gpu) {
  m_id = sub_partition_id;
  m_config = config;
  m_stats = stats;
  m_gpu = gpu;
  m_memcpy_cycle_offset = 0;

  assert(m_id < m_config->m_n_mem_sub_partition);

  char L2c_name[32];
  snprintf(L2c_name, 32, "L2_bank_%03d", m_id);
  m_L2interface = new L2interface(this);
  m_mf_allocator = new partition_mf_allocator(config);

  if (!m_config->m_L2_config.disabled())
    m_L2cache = new l2_cache(L2c_name, m_config->m_L2_config, -1, -1,
                             m_L2interface, m_mf_allocator,
                             IN_PARTITION_L2_MISS_QUEUE, gpu, L2_GPU_CACHE);

  unsigned int icnt_L2;
  unsigned int L2_dram;
  unsigned int dram_L2;
  unsigned int L2_icnt;
  sscanf(m_config->gpgpu_L2_queue_config, "%u:%u:%u:%u", &icnt_L2, &L2_dram,
         &dram_L2, &L2_icnt);
  m_icnt_L2_queue = new fifo_pipeline<mem_fetch>("icnt-to-L2", 0, icnt_L2);
  m_L2_dram_queue = new fifo_pipeline<mem_fetch>("L2-to-dram", 0, L2_dram);
  m_dram_L2_queue = new fifo_pipeline<mem_fetch>("dram-to-L2", 0, dram_L2);
  m_L2_icnt_queue = new fifo_pipeline<mem_fetch>("L2-to-icnt", 0, L2_icnt);
  print_flag = 0;
  wb_addr = -1;
}

memory_sub_partition::~memory_sub_partition() {
  delete m_icnt_L2_queue;
  delete m_L2_dram_queue;
  delete m_dram_L2_queue;
  delete m_L2_icnt_queue;
  delete m_L2cache;
  delete m_L2interface;
}

void memory_sub_partition::cache_cycle(unsigned cycle) {
  // L2 fill responses
  if (!m_config->m_L2_config.disabled()) {
    if (m_L2cache->access_ready() && !m_L2_icnt_queue->full()) {
      mem_fetch *mf = m_L2cache->next_access();
      if (mf->get_access_type() !=
          L2_WR_ALLOC_R) {  // Don't pass write allocate read request back to
                            // upper level cache
        mf->set_reply();
        mf->set_status(IN_PARTITION_L2_TO_ICNT_QUEUE,
                       m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
        m_L2_icnt_queue->push(mf);
      } else {
        if (m_config->m_L2_config.m_write_alloc_policy == FETCH_ON_WRITE) {
          mem_fetch *original_wr_mf = mf->get_original_wr_mf();
          assert(original_wr_mf);
          original_wr_mf->set_reply();
          original_wr_mf->set_status(
              IN_PARTITION_L2_TO_ICNT_QUEUE,
              m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
          m_L2_icnt_queue->push(original_wr_mf);
        }
        m_request_tracker.erase(mf);
        delete mf;
      }
    }
  }

  // DRAM to L2 (texture) and icnt (not texture)
  if (!m_dram_L2_queue->empty()) {
    mem_fetch *mf = m_dram_L2_queue->top();
  
    if (!m_config->m_L2_config.disabled() && m_L2cache->waiting_for_fill(mf)) {
      if (m_L2cache->fill_port_free()) {
        mf->set_status(IN_PARTITION_L2_FILL_QUEUE,
                       m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
        m_L2cache->fill(mf, m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle +
                                m_memcpy_cycle_offset);
        m_dram_L2_queue->pop();      
      }
    } else if (!m_L2_icnt_queue->full()) {
      if (mf->is_write() && mf->get_type() == WRITE_ACK)
        mf->set_status(IN_PARTITION_L2_TO_ICNT_QUEUE,
                       m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
      m_L2_icnt_queue->push(mf);
      m_dram_L2_queue->pop();
    }
  }

  // prior L2 misses inserted into m_L2_dram_queue here
  if (!m_config->m_L2_config.disabled()) m_L2cache->cycle();

  // new L2 texture accesses and/or non-texture accesses
  if (!m_L2_dram_queue->full() && !m_icnt_L2_queue->empty()) {
    mem_fetch *mf = m_icnt_L2_queue->top();
    if (!m_config->m_L2_config.disabled() &&
        ((m_config->m_L2_texure_only && mf->istexture()) ||
         (!m_config->m_L2_texure_only))) {
      // L2 is enabled and access is for L2
      bool output_full = m_L2_icnt_queue->full();
      bool port_free = m_L2cache->data_port_free();
      if (!output_full && port_free) {
        std::list<cache_event> events;
        enum cache_request_status status =
            m_L2cache->access(mf->get_addr(), mf,
                              m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle +
                                  m_memcpy_cycle_offset,
                              events);
        bool write_sent = was_write_sent(events);
        bool read_sent = was_read_sent(events);
        MEM_SUBPART_DPRINTF("Probing L2 cache Address=%llx, status=%u\n",
                            mf->get_addr(), status);

        if (status == HIT) {
          if (!write_sent) {
            // L2 cache replies
            assert(!read_sent);
            if (mf->get_access_type() == L1_WRBK_ACC) {
              m_request_tracker.erase(mf);
              delete mf;
            } else {
              mf->set_reply();
              mf->set_status(IN_PARTITION_L2_TO_ICNT_QUEUE,
                             m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
              m_L2_icnt_queue->push(mf);
            }
            m_icnt_L2_queue->pop();
          } else {
            assert(write_sent);
            m_icnt_L2_queue->pop();
          }
        } else if (status != RESERVATION_FAIL) {
          if (mf->is_write() &&
              (m_config->m_L2_config.m_write_alloc_policy == FETCH_ON_WRITE ||
               m_config->m_L2_config.m_write_alloc_policy ==
                   LAZY_FETCH_ON_READ) &&
              !was_writeallocate_sent(events)) {
            if (mf->get_access_type() == L1_WRBK_ACC) {
              m_request_tracker.erase(mf);
              delete mf;
            } else if (m_config->m_L2_config.get_write_policy() == WRITE_BACK) {
              mf->set_reply();
              mf->set_status(IN_PARTITION_L2_TO_ICNT_QUEUE,
                             m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
              m_L2_icnt_queue->push(mf);
            }
          }
          // L2 cache accepted request
          m_icnt_L2_queue->pop();
        } else {
          assert(!write_sent);
          assert(!read_sent);
          // L2 cache lock-up: will try again next cycle
        }
      }
    } else {
      // L2 is disabled or non-texture access to texture-only L2
      mf->set_status(IN_PARTITION_L2_TO_DRAM_QUEUE,
                     m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
      m_L2_dram_queue->push(mf);
      m_icnt_L2_queue->pop();
    }
  }

  // ROP delay queue
  if (!m_rop.empty() && (cycle >= m_rop.front().ready_cycle) &&
      !m_icnt_L2_queue->full()) {
    mem_fetch *mf = m_rop.front().req;
    m_rop.pop();
    m_icnt_L2_queue->push(mf);
    mf->set_status(IN_PARTITION_ICNT_TO_L2_QUEUE,
                   m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
  }

  // //tracker
  // if(m_gpu->gpu_sim_cycle > 999) {
  //   if((m_gpu->gpu_sim_cycle % 5000) == 0) {
  //     if((print_flag == 0)) {
  //       printf("Memory Sub Parition %u: pending memory requests:\n", m_id);
  //       if (!m_request_tracker.empty()) {
  //         for (std::set<mem_fetch *>::const_iterator r = m_request_tracker.begin();
  //             r != m_request_tracker.end(); ++r) {
  //           mem_fetch *mf = *r;
  //           if (mf)
  //             mf->print(stdout);
  //           else
  //             printf(" <NULL mem_fetch?>\n");
  //         }
  //       } else {
  //         printf("Memory Sub Parition %u: none-pending memory requests:\n", m_id);
  //       }
  //       print_flag = 1;
  //     } 
  //   } else if(((m_gpu->gpu_sim_cycle % 5000) == 1)) {
  //     if((print_flag == 0)) {
  //       printf("Memory Sub Parition %u: pending memory requests:\n", m_id);
  //       if (!m_request_tracker.empty()) {
  //         for (std::set<mem_fetch *>::const_iterator r = m_request_tracker.begin();
  //             r != m_request_tracker.end(); ++r) {
  //           mem_fetch *mf = *r;
  //           if (mf)
  //             mf->print(stdout);
  //           else
  //             printf(" <NULL mem_fetch?>\n");
  //         }
  //       } else {
  //         printf("Memory Sub Parition %u: none-pending memory requests:\n", m_id);
  //       }
  //     } else {
  //       print_flag = 0;
  //     }
  //   }
  // // } pshyun 
  // }

}

bool memory_sub_partition::full() const { return m_icnt_L2_queue->full(); }

bool memory_sub_partition::full(unsigned size) const {
  return m_icnt_L2_queue->is_avilable_size(size);
}

bool memory_sub_partition::L2_dram_queue_empty() const {
  return m_L2_dram_queue->empty();
}

class mem_fetch *memory_sub_partition::L2_dram_queue_top() const {
  return m_L2_dram_queue->top();
}

void memory_sub_partition::L2_dram_queue_pop() { m_L2_dram_queue->pop(); }

bool memory_sub_partition::dram_L2_queue_full() const {
  return m_dram_L2_queue->full();
}

void memory_sub_partition::dram_L2_queue_push(class mem_fetch *mf) {
  m_dram_L2_queue->push(mf);
}

// pshyun {
bool memory_partition_unit::ndc_fill_queue_empty() const
{
   return m_ndc_fill_queue->empty(); 
}

class mem_fetch* memory_partition_unit::ndc_fill_queue_top() const
{
   return m_ndc_fill_queue->top(); 
}

void memory_partition_unit::ndc_fill_queue_pop() 
{
   m_ndc_fill_queue->pop(); 
}

void memory_partition_unit::ndc_fill_queue_push( class mem_fetch* mf ) 
{
   m_ndc_fill_queue->push(mf); 
}

bool memory_partition_unit::ndc_fill_queue_full() const
{
   return m_ndc_fill_queue->full(); 
}

bool memory_partition_unit::dram_cxl_queue_empty() const
{
   return m_dram_cxl_queue->empty(); 
}
bool memory_partition_unit::dram_cxl_queue_full() const
{
   return m_dram_cxl_queue->full(); 
}
void memory_partition_unit::dram_cxl_queue_push( class mem_fetch* mf )
{
   m_dram_cxl_queue->push(mf); 
}
class mem_fetch* memory_partition_unit::dram_cxl_queue_top() const
{
   return m_dram_cxl_queue->top(); 
}
void memory_partition_unit::dram_cxl_queue_pop() 
{
   m_dram_cxl_queue->pop(); 
}

std::vector<mem_fetch *> memory_partition_unit::breakdown_wrbk_request_to_sector_requests(mem_fetch *mf) {
  std::vector<mem_fetch *> result;
  mem_access_sector_mask_t sector_mask = mf->get_access_sector_mask();
  if (mf->get_data_size() == SECTOR_SIZE &&
      mf->get_access_sector_mask().count() == 1) {
    result.push_back(mf);
  }
  else if (mf->get_data_size() == MAX_MEMORY_ACCESS_SIZE) {
    // break down every sector
    mem_access_byte_mask_t mask;
    for (unsigned i = 0; i < SECTOR_CHUNCK_SIZE; i++) {
      for (unsigned k = i * SECTOR_SIZE; k < (i + 1) * SECTOR_SIZE; k++) {
        mask.set(k);
      }
      mem_fetch *n_mf = m_wrbk_mf_allocator->alloc(
          mf->get_addr() + SECTOR_SIZE * i, mf->get_access_type(),
          mf->get_access_warp_mask(), mf->get_access_byte_mask() & mask,
          std::bitset<SECTOR_CHUNCK_SIZE>().set(i), SECTOR_SIZE, mf->is_write(),
          m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle, mf->get_wid(),
          mf->get_sid(), mf->get_tpc(), mf, mf->get_streamID());

      result.push_back(n_mf);
    }
    
    wrbk_access_cnt++;
  } else {
    for (unsigned i = 0; i < SECTOR_CHUNCK_SIZE; i++) {
      if (sector_mask.test(i)) {
        mem_access_byte_mask_t mask;
        for (unsigned k = i * SECTOR_SIZE; k < (i + 1) * SECTOR_SIZE; k++) {
          mask.set(k);
        }
        mem_fetch *n_mf = m_wrbk_mf_allocator->alloc(
            mf->get_addr() + SECTOR_SIZE * i, mf->get_access_type(),
            mf->get_access_warp_mask(), mf->get_access_byte_mask() & mask,
            std::bitset<SECTOR_CHUNCK_SIZE>().set(i), SECTOR_SIZE,
            mf->is_write(), m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle,
            mf->get_wid(), mf->get_sid(), mf->get_tpc(), mf,
            mf->get_streamID());

        result.push_back(n_mf);
      }
    }
    
    wrbk_access_cnt++;
  }
  return result;
}



// } pshyun

void memory_sub_partition::print_cache_stat(unsigned &accesses,
                                            unsigned &misses) const {
  FILE *fp = stdout;
  if (!m_config->m_L2_config.disabled()) m_L2cache->print(fp, accesses, misses);
}

void memory_sub_partition::print(FILE *fp) const {
  if (!m_request_tracker.empty()) {
    fprintf(fp, "Memory Sub Parition %u: pending memory requests:\n", m_id);
    for (std::set<mem_fetch *>::const_iterator r = m_request_tracker.begin();
         r != m_request_tracker.end(); ++r) {
      mem_fetch *mf = *r;
      if (mf)
        mf->print(fp);
      else
        fprintf(fp, " <NULL mem_fetch?>\n");
    }
  }
  if (!m_config->m_L2_config.disabled()) m_L2cache->display_state(fp);
}

void memory_stats_t::visualizer_print(gzFile visualizer_file) {
  gzprintf(visualizer_file, "Ltwowritemiss: %d\n", L2_write_miss);
  gzprintf(visualizer_file, "Ltwowritehit: %d\n", L2_write_hit);
  gzprintf(visualizer_file, "Ltworeadmiss: %d\n", L2_read_miss);
  gzprintf(visualizer_file, "Ltworeadhit: %d\n", L2_read_hit);
  clear_L2_stats_pw();

  if (num_mfs)
    gzprintf(visualizer_file, "averagemflatency: %lld\n",
             mf_total_lat / num_mfs);
}

void memory_stats_t::clear_L2_stats_pw() {
  L2_write_miss = 0;
  L2_write_hit = 0;
  L2_read_miss = 0;
  L2_read_hit = 0;
}

void gpgpu_sim::print_dram_stats(FILE *fout) const {
  unsigned cmd = 0;
  unsigned activity = 0;
  unsigned nop = 0;
  unsigned act = 0;
  unsigned pre = 0;
  unsigned rd = 0;
  unsigned wr = 0;
  unsigned wr_WB = 0;
  unsigned req = 0;
  unsigned tot_cmd = 0;
  unsigned tot_nop = 0;
  unsigned tot_act = 0;
  unsigned tot_pre = 0;
  unsigned tot_rd = 0;
  unsigned tot_wr = 0;
  unsigned tot_req = 0;

  for (unsigned i = 0; i < m_memory_config->m_n_mem; i++) {
    m_memory_partition_unit[i]->set_dram_power_stats(cmd, activity, nop, act,
                                                     pre, rd, wr, wr_WB, req);
    tot_cmd += cmd;
    tot_nop += nop;
    tot_act += act;
    tot_pre += pre;
    tot_rd += rd;
    tot_wr += wr + wr_WB;
    tot_req += req;
  }
  fprintf(fout, "gpgpu_n_dram_reads = %d\n", tot_rd);
  fprintf(fout, "gpgpu_n_dram_writes = %d\n", tot_wr);
  fprintf(fout, "gpgpu_n_dram_activate = %d\n", tot_act);
  fprintf(fout, "gpgpu_n_dram_commands = %d\n", tot_cmd);
  fprintf(fout, "gpgpu_n_dram_noops = %d\n", tot_nop);
  fprintf(fout, "gpgpu_n_dram_precharges = %d\n", tot_pre);
  fprintf(fout, "gpgpu_n_dram_requests = %d\n", tot_req);
}

unsigned memory_sub_partition::flushL2() {
  if (!m_config->m_L2_config.disabled()) {
    m_L2cache->flush();
  }
  return 0;  // TODO: write the flushed data to the main memory
}

unsigned memory_sub_partition::invalidateL2() {
  if (!m_config->m_L2_config.disabled()) {
    m_L2cache->invalidate();
  }
  return 0;
}

bool memory_sub_partition::busy() const { return !m_request_tracker.empty(); }

std::vector<mem_fetch *>
memory_sub_partition::breakdown_request_to_sector_requests(mem_fetch *mf) {
  std::vector<mem_fetch *> result;
  mem_access_sector_mask_t sector_mask = mf->get_access_sector_mask();
  if (mf->get_data_size() == SECTOR_SIZE &&
      mf->get_access_sector_mask().count() == 1) {
    result.push_back(mf);
  } else if (mf->get_data_size() == MAX_MEMORY_ACCESS_SIZE) {
    // break down every sector
    mem_access_byte_mask_t mask;
    for (unsigned i = 0; i < SECTOR_CHUNCK_SIZE; i++) {
      for (unsigned k = i * SECTOR_SIZE; k < (i + 1) * SECTOR_SIZE; k++) {
        mask.set(k);
      }
      mem_fetch *n_mf = m_mf_allocator->alloc(
          mf->get_addr() + SECTOR_SIZE * i, mf->get_access_type(),
          mf->get_access_warp_mask(), mf->get_access_byte_mask() & mask,
          std::bitset<SECTOR_CHUNCK_SIZE>().set(i), SECTOR_SIZE, mf->is_write(),
          m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle, mf->get_wid(),
          mf->get_sid(), mf->get_tpc(), mf, mf->get_streamID());

      result.push_back(n_mf);
    }
    // This is for constant cache
  } else if (mf->get_data_size() == 64 &&
             (mf->get_access_sector_mask().all() ||
              mf->get_access_sector_mask().none())) {
    unsigned start;
    if (mf->get_addr() % MAX_MEMORY_ACCESS_SIZE == 0)
      start = 0;
    else
      start = 2;
    mem_access_byte_mask_t mask;
    for (unsigned i = start; i < start + 2; i++) {
      for (unsigned k = i * SECTOR_SIZE; k < (i + 1) * SECTOR_SIZE; k++) {
        mask.set(k);
      }
      mem_fetch *n_mf = m_mf_allocator->alloc(
          mf->get_addr(), mf->get_access_type(), mf->get_access_warp_mask(),
          mf->get_access_byte_mask() & mask,
          std::bitset<SECTOR_CHUNCK_SIZE>().set(i), SECTOR_SIZE, mf->is_write(),
          m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle, mf->get_wid(),
          mf->get_sid(), mf->get_tpc(), mf, mf->get_streamID());

      result.push_back(n_mf);
    }
  } else {
    // break down every sector
    for (unsigned i = 0; i < SECTOR_CHUNCK_SIZE; i++) {
      if (sector_mask.test(i)) {
        mem_access_byte_mask_t mask;
        for (unsigned k = i * SECTOR_SIZE; k < (i + 1) * SECTOR_SIZE; k++) {
          mask.set(k);
        }
        mem_fetch *n_mf = m_mf_allocator->alloc(
            mf->get_addr() + SECTOR_SIZE * i, mf->get_access_type(),
            mf->get_access_warp_mask(), mf->get_access_byte_mask() & mask,
            std::bitset<SECTOR_CHUNCK_SIZE>().set(i), SECTOR_SIZE,
            mf->is_write(), m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle,
            mf->get_wid(), mf->get_sid(), mf->get_tpc(), mf,
            mf->get_streamID());

        result.push_back(n_mf);
      }
    }
  }
  if (result.size() == 0) assert(0 && "no mf sent");
  return result;
}

void memory_sub_partition::push(mem_fetch *m_req, unsigned long long cycle) {
  if (m_req) {
    m_stats->memlatstat_icnt2mem_pop(m_req);
    std::vector<mem_fetch *> reqs;
    if (m_config->m_L2_config.m_cache_type == SECTOR)
      reqs = breakdown_request_to_sector_requests(m_req);
    else
      reqs.push_back(m_req);

    for (unsigned i = 0; i < reqs.size(); ++i) {
      mem_fetch *req = reqs[i];
      m_request_tracker.insert(req);
      if (req->istexture()) {
        m_icnt_L2_queue->push(req);
        req->set_status(IN_PARTITION_ICNT_TO_L2_QUEUE,
                        m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
      } else {
        rop_delay_t r;
        r.req = req;
        r.ready_cycle = cycle + m_config->rop_latency;
        m_rop.push(r);
        req->set_status(IN_PARTITION_ROP_DELAY,
                        m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
      }
    }
  }
}

mem_fetch *memory_sub_partition::pop() {
  mem_fetch *mf = m_L2_icnt_queue->pop();
  m_request_tracker.erase(mf);
  if (mf && mf->isatomic()) mf->do_atomic();
  if (mf && (mf->get_access_type() == L2_WRBK_ACC ||
             mf->get_access_type() == L1_WRBK_ACC)) {
    delete mf;
    mf = NULL;
  }
  return mf;
}

mem_fetch *memory_sub_partition::top() {
  mem_fetch *mf = m_L2_icnt_queue->top();
  if (mf && (mf->get_access_type() == L2_WRBK_ACC ||
             mf->get_access_type() == L1_WRBK_ACC)) {
    m_L2_icnt_queue->pop();
    m_request_tracker.erase(mf);
    delete mf;
    mf = NULL;
  }
  return mf;
}

void memory_sub_partition::set_done(mem_fetch *mf) {
  m_request_tracker.erase(mf);
}

void memory_sub_partition::accumulate_L2cache_stats(
    class cache_stats &l2_stats) const {
  if (!m_config->m_L2_config.disabled()) {
    l2_stats += m_L2cache->get_stats();
  }
}

void memory_sub_partition::get_L2cache_sub_stats(
    struct cache_sub_stats &css) const {
  if (!m_config->m_L2_config.disabled()) {
    m_L2cache->get_sub_stats(css);
  }
}

void memory_sub_partition::get_L2cache_sub_stats_pw(
    struct cache_sub_stats_pw &css) const {
  if (!m_config->m_L2_config.disabled()) {
    m_L2cache->get_sub_stats_pw(css);
  }
}

void memory_sub_partition::clear_L2cache_stats_pw() {
  if (!m_config->m_L2_config.disabled()) {
    m_L2cache->clear_pw();
  }
}

void memory_sub_partition::visualizer_print(gzFile visualizer_file) {
  // Support for L2 AerialVision stats
  // Per-sub-partition stats would be trivial to extend from this
  cache_sub_stats_pw temp_sub_stats;
  get_L2cache_sub_stats_pw(temp_sub_stats);

  m_stats->L2_read_miss += temp_sub_stats.read_misses;
  m_stats->L2_write_miss += temp_sub_stats.write_misses;
  m_stats->L2_read_hit += temp_sub_stats.read_hits;
  m_stats->L2_write_hit += temp_sub_stats.write_hits;

  clear_L2cache_stats_pw();
}

// pshyun {
class mem_fetch * memory_partition_unit::find_orig_mf(class mem_fetch *mf) {
    mf_map::iterator e = mf_map_rvs.find(mf);
    mem_fetch *mf_orig = nullptr;
    if (e != mf_map_rvs.end())
        mf_orig = mf_map_rvs[mf];
    return mf_orig;
}
class mem_fetch * memory_partition_unit::create_new_mf(class mem_fetch *mf) {
    mem_fetch *mf_new;
    mf->set_l2_done(false);
    mf->set_dram_done(false);
    mf->set_cxl_req_type(CXL_INVALID);
    mf->set_cxl_ret_path(CXL_NONE);
    mf->set_ndc_resp(NDC_INVALID);
    mf_new = new mem_fetch(*mf);
    mf_map_rvs[mf_new] = mf;       // set mf_map

//YH_DEBUG:#ifdef YH_DEBUG
    if ((mf->get_request_uid() == TGT_UID) || (mf_new->get_request_uid() == TGT_UID)) {
        // if ((gpu_sim_cycle >= LOG_ST) && (gpu_sim_cycle <= LOG_ED)) {
        printf("[YH_DEBUG][%d][mp%d][create_new_mf] mf_orig.uid: %d, mf_new.uid: %d\n", m_gpu->gpu_sim_cycle, m_id, mf->get_request_uid(), mf_new->get_request_uid());
        mf->print(stdout);
        mf_new->print(stdout);
        //}
    }
//YH_DEBUG:#endif // YH_DEBUG
    //TODO:printf("[YH_DEBUG][%d][mp%d][dram_acc_tracker] time:%d, addr: 0x%x\n", gpu_sim_cycle, m_id, gpu_sim_cycle, mf->get_addr()); // dram access tracker...
    m_request_tracker_dram.insert(mf_new);
    return mf_new;
}


void memory_partition_unit::delete_new_mf(class mem_fetch *mf) {
    mem_fetch *mf_orig;
    mf_orig = find_orig_mf(mf);
    if (mf_orig != nullptr) {
        if (mf->get_l2_done() && mf->get_dram_done()){
            //printf("[YH_DEBUG][%d][mp%d][delete_new_mf] mf.uid: %d\n", m_gpu->gpu_sim_cycle, m_id, mf->get_request_uid());
            mf_map_rvs.erase(mf);
//YH_DEBUG:#ifdef YH_DEBUG
            if ((mf->get_request_uid() == TGT_UID) || (mf_orig->get_request_uid() == TGT_UID)) {
                // if ((gpu_sim_cycle >= LOG_ST) && (gpu_sim_cycle <= LOG_ED)) {
                //printf("[YH_DEBUG][%d][mp%d][delete_new_mf] mf.uid: %d\n", m_gpu->gpu_sim_cycle, m_id, mf->get_request_uid());
                //mf->print(stdout);
                //}
            }
//YH_DEBUG:#endif  // YH_DEBUG
            if (m_request_tracker_dram.find(mf) != m_request_tracker_dram.end()) {
                m_request_tracker_dram.erase(mf);
                if(mf->get_ndc_resp() == NDC_HIT || mf->get_ndc_resp() == NDC_MISS) delete mf;
                //delete mf;  // remove new mf (l2_done and dram_done) // FIXME
            } else {
                //printf("[YH_DEBUG][%d][mp%d][delete_new_mf] mf.uid: %d. it is not in the request tracker, but try to delete.\n", m_gpu->gpu_sim_cycle, m_id, mf->get_request_uid());
                //mf->print(stdout);
                assert(0);
                delete mf;  // remove new mf (l2_done and dram_done)
            }
        }
    } else {
#ifdef YH_DEBUG
        if (mf->get_request_uid() == TGT_UID) {
            if ((m_gpu->gpu_sim_cycle >= LOG_ST) && (m_gpu->gpu_sim_cycle <= LOG_ED)) {
                //printf("[YH_DEBUG][%d][mp%d][delete_new_mf][for NDC_WR_ALLOC_R or NDC_WB] mf.uid: %d\n", m_gpu->gpu_sim_cycle, m_id, mf->get_request_uid());
                //mf->print(stdout);
            }
        }
#endif  // YH_DEBUG
        //printf("[YH_DEBUG][%d][mp%d][delete_new_mf] mf.uid: %d\n", m_gpu->gpu_sim_cycle, m_id, mf->get_request_uid());
        if (m_request_tracker_ndc.find(mf) != m_request_tracker_ndc.end()) {
            if (m_request_tracker_dram.find(mf) != m_request_tracker_dram.end()) {
                //printf("[YH_DEBUG][%d][mp%d][delete_new_mf] mf.uid: %d. it is in the request tracker_dram. why?.\n", m_gpu->gpu_sim_cycle, m_id, mf->get_request_uid());
                //mf->print(stdout);
                assert(0);
                delete mf;  // remove new mf (l2_done and dram_done)
            }
            m_request_tracker_ndc.erase(mf);
            delete mf;  // remove new mf from NDC WR_LINE_FILL
        } else {
            //printf("[YH_DEBUG][%d][mp%d][delete_new_mf] mf.uid: %d. it is not in the request tracker_ndc, but try to delete.\n", m_gpu->gpu_sim_cycle, m_id, mf->get_request_uid());
            //mf->print(stdout);
            //assert(0); //FIXME
            delete mf;  // remove new mf (l2_done and dram_done)
        }
    }
}

void memory_partition_unit::print_returnq() const {
    printf("[YH_DEBUG][%d][mp%d] Return Queue Information\n", m_gpu->gpu_sim_cycle, m_id);
    printf("[YH_DEBUG][%d][mp%d] Return Queue Size: %d\n", m_gpu->gpu_sim_cycle, m_id, m_dram->returnq->get_length());
    m_dram->print(stdout);
}
// } pshyun