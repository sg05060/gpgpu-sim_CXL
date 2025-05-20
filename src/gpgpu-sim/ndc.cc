// Copyright (c) 2009-2011, Tor M. Aamodt, Wilson W.L. Fung, Ali Bakhoda,
// Ivan Sham, George L. Yuan,
// The University of British Columbia
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice, this
// list of conditions and the following disclaimer in the documentation and/or
// other materials provided with the distribution.
// Neither the name of The University of British Columbia nor the names of its
// contributors may be used to endorse or promote products derived from this
// software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "gpu-sim.h"
#include "gpu-misc.h"
#include "dram.h"
#include "mem_latency_stat.h"
#include "dram_sched.h"
#include "mem_fetch.h"
#include "l2cache.h"
#include "ndc.h"

#ifdef DRAM_VERIFY
int PRINT_CYCLE = 0;
#endif

template class fifo_pipeline<mem_fetch>;
template class fifo_pipeline<dram_req_t>;

// pshyun {
mem_fetch *ndc_mf_allocator::alloc(new_addr_type addr,
                                         mem_access_type type, unsigned size,
                                         bool wr, unsigned long long cycle,
                                         unsigned long long streamID) const {
  assert(wr);
  mem_access_t access(type, addr, size, wr, m_memory_config->gpgpu_ctx);
  mem_fetch *mf = new mem_fetch(access, NULL, streamID, WRITE_PACKET_SIZE, -1,
                                -1, -1, m_memory_config, cycle);
  return mf;
}

mem_fetch *ndc_mf_allocator::alloc(
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
// } pshyun

ndc_t::~ndc_t() {
   if (!m_config->m_L3_NDC_config.disabled()) {
      delete m_ndc_cache;
      delete m_L3_NDC_interface;
   }
}

// pshyun : add *gpu argument
ndc_t::ndc_t(unsigned int partition_id, const struct memory_config *config, memory_stats_t *stats, memory_partition_unit *mp, gpgpu_sim *gpu)
    : dram_t(partition_id, config, stats, mp, gpu)
{
   // error threshold
   int err_cnt = 0;

   id = partition_id;
   m_memory_partition_unit = mp;
   m_stats = stats;
   m_config = config;
   m_gpu = gpu;

   if (!m_config->m_L3_NDC_config.disabled())
   {
      m_L3_NDC_interface = new L3_NDC_interface(this);
      m_mf_allocator = new ndc_mf_allocator(m_config);
      // m_memport = m_L3_NDC_interface
      // At m_ndc_cache->cycle(), m_miss_queue.pop_front() and memport.push(mf) = m_L3_NDC_interface.push(mf) = returnq.push(mf)
      m_ndc_cache = new l3_ndc_cache("NDC", config->m_L3_NDC_config, -1, -1, m_L3_NDC_interface, m_mf_allocator, IN_PARTITION_NDC_CACHE_QUEUE, gpu, OTHER_GPU_CACHE);
   }


   // rowblp
   access_num = 0;
   hits_num = 0;
   read_num = 0;
   write_num = 0;
   hits_read_num = 0;
   hits_write_num = 0;
   banks_1time = 0;
   banks_acess_total = 0;
   banks_acess_total_after = 0;
   banks_time_ready = 0;
   banks_access_ready_total = 0;
   issued_two = 0;
   issued_total = 0;
   issued_total_row = 0;
   issued_total_col = 0; 

   CCDc = 0;
   RRDc = 0;
   RTWc = 0;
   WTRc = 0;
   
   wasted_bw_row = 0;
   wasted_bw_col = 0;
   util_bw = 0;
   idle_bw = 0;
   RCDc_limit = 0;
   CCDLc_limit = 0;
   CCDLc_limit_alone = 0;
   CCDc_limit = 0;
   WTRc_limit = 0;
   WTRc_limit_alone = 0;
   RCDWRc_limit = 0;
   RTWc_limit = 0;
   RTWc_limit_alone = 0;
   rwq_limit = 0;
   write_to_read_ratio_blp_rw_average = 0;
   bkgrp_parallsim_rw = 0;

   rw = READ; //read mode is default

	bkgrp = (bankgrp_t**) calloc(sizeof(bankgrp_t*), m_config->nbkgrp);
	bkgrp[0] = (bankgrp_t*) calloc(sizeof(bank_t), m_config->nbkgrp);
	for (unsigned i=1; i<m_config->nbkgrp; i++) {
		bkgrp[i] = bkgrp[0] + i;
	}
	for (unsigned i=0; i<m_config->nbkgrp; i++) {
		bkgrp[i]->CCDLc = 0;
		bkgrp[i]->RTPLc = 0;
	}

   bk = (bank_t**) calloc(sizeof(bank_t*),m_config->nbk);
   bk[0] = (bank_t*) calloc(sizeof(bank_t),m_config->nbk);
   for (unsigned i=1;i<m_config->nbk;i++) 
      bk[i] = bk[0] + i;
   for (unsigned i=0;i<m_config->nbk;i++) {
      bk[i]->state = BANK_IDLE;
      bk[i]->bkgrpindex = i/(m_config->nbk/m_config->nbkgrp);
   }
   prio = 0;  
   rwq = new fifo_pipeline<dram_req_t>("rwq",m_config->CL,m_config->CL+1);
   mrqq = new fifo_pipeline<dram_req_t>("mrqq",0,2);
   // yhyang {
   fill_mrqq = new fifo_pipeline<dram_req_t>("fill_mrqq", 0, 2);
   dram_ndc_fill_queue = new fifo_pipeline<dram_req_t>("dram_ndc_fill_queue", 0, 16);  // assume enough queue size. if not, increase...
   ndc_rsv_fail_queue = new fifo_pipeline<dram_req_t>("ndc_rsv_fail_queue", 0, 32);
   // } yhyang
   returnq = new fifo_pipeline<mem_fetch>("dramreturnq",0,m_config->gpgpu_dram_return_queue_size==0?1024:m_config->gpgpu_dram_return_queue_size); 
   m_frfcfs_scheduler = NULL;
   if ( m_config->scheduler_type == DRAM_FRFCFS )
      m_frfcfs_scheduler = new frfcfs_scheduler(m_config,this,stats);
   n_cmd = 0;
   n_activity = 0;
   n_nop = 0; 
   n_act = 0; 
   n_pre = 0; 
   n_rd = 0;
   n_wr = 0;
   n_req = 0;
   max_mrqs_temp = 0;
   bwutil = 0;
   max_mrqs = 0;
   ave_mrqs = 0;

   for (unsigned i=0;i<10;i++) {
      dram_util_bins[i]=0;
      dram_eff_bins[i]=0;
   }
   last_n_cmd = last_n_activity = last_bwutil = 0;

   n_cmd_partial = 0;
   n_activity_partial = 0;
   n_nop_partial = 0;  
   n_act_partial = 0;  
   n_pre_partial = 0;  
   n_req_partial = 0;
   ave_mrqs_partial = 0;
   bwutil_partial = 0;

   if ( queue_limit() )
      mrqq_Dist = StatCreate("mrqq_length",1, queue_limit());
   else //queue length is unlimited; 
      mrqq_Dist = StatCreate("mrqq_length",1,64); //track up to 64 entries
}



#define DEC2ZERO(x) x = (x)? (x-1) : 0;
#define SWAP(a,b) a ^= b; b ^= a; a ^= b;

void ndc_t::cache_cycle(/*unsigned cycle*/) {
    // NDC fill responses
    if (!m_config->m_L3_NDC_config.disabled()) {
        if (m_ndc_cache->access_ready() && !returnq_full()) {
            mem_fetch *mf = m_ndc_cache->next_access();  // mf is filled request

            if ((mf->get_access_type() == NDC_WR_ALLOC_R) && (mf->get_cxl_ret_path() == CXL_NONE)) {
                
                // debug
                printf("[PSH_DEBUG]Write MSHR-MISS MSHR-HIT Response Here...\n"); 
                mf->print(stdout);
                
                mf->set_status(IN_PARTITION_MC_RETURNQ, m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
                mf->set_ndc_resp(NDC_FILL_DONE);
                returnq->push(mf);
                //delete mf;
                
                // NDC write -> miss -> mshr hit -> fill done. does not need to return to memory partition unit
                if (mf->get_request_uid() == TGT_UID) {
                    printf("[YH_DEBUG][%d][NDC%d][fill_queue->no return_queue] fill_done_queue to no returnq. uid = %d, addr = 0x%x, status = %d\n", m_gpu->gpu_sim_cycle, id, mf->get_request_uid(),
                           mf->get_addr(), mf->get_status());
                    mf->print(stdout);
                }
                //printf("[PSH_DEBUG][Check Delete0]\n");
                //delete mf;
            } else {
                if((mf->get_access_type()==NDC_LINEFILL_W) && (mf->get_cxl_ret_path() == CXL_DRAM)) { printf("[PSH_DEBUG]Write MSHR-MISS MSHR-MISS Response Here...\n"); mf->print(stdout);}
                mf->set_status(IN_PARTITION_MC_RETURNQ, m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
                mf->set_ndc_resp(NDC_FILL_DONE);
                // YH_DEBUG:#ifdef YH_DEBUG
                if (mf->get_request_uid() == TGT_UID) {
                    printf("[YH_DEBUG][%d][NDC%d][fill_queue->return_queue] fill_done_queue to returnq. uid = %d, addr = 0x%x, status = %d\n", m_gpu->gpu_sim_cycle, id, mf->get_request_uid(), mf->get_addr(),
                           mf->get_status());
                    mf->print(stdout);
                }
                // YH_DEBUG:#endif // YH_DEBUG
                returnq->push(mf);
#ifdef YH_DEBUG
                if ((m_gpu->gpu_sim_cycle >= LOG_ST) && (m_gpu->gpu_sim_cycle <= LOG_ED)) {
                    if (id == 10) {
                        printf("[YH_DEBUG][%d][NDC%d][fill_done] fill_done_queue to returnq. uid = %d, addr = 0x%x, status = %d\n", m_gpu->gpu_sim_cycle, id, mf->get_request_uid(), mf->get_addr(),
                               mf->get_status());
                        mf->print(stdout);
                    }
                }
#endif  // YH_DEBUG
            }
        }
    }

    // prior L2 misses inserted into m_L2_dram_queue here
    if (!m_config->m_L3_NDC_config.disabled())
        if (!returnq_full()) {
            m_ndc_cache->cycle();  // miss_queue to returnq..... hmmmmmmmm. NDC_WR_ALLOC_R request will be transfered to returnq here.
                                   // also evicted data will be transfered to returnq
        }
}

void ndc_t::ndc_access(dram_req_t *cmd, bool from_rsvq) {
   bool rm_cmd_flag = true;
   mem_fetch *data = cmd->data;
   if (cmd->data->get_request_uid() == TGT_UID) {
       printf("[YH_DEBUG][uid:%d] For NDC access..\n", cmd->data->get_request_uid());
       if (err_cnt++ > 1000) {
           printf("[YH_DEBUG][uid:%d] For NDC access same request is keep try to access NDC\n", cmd->data->get_request_uid());
           assert(0);
       }
   }

#ifdef YH_DEBUG
   if (cmd->data->get_request_uid() == TGT_UID) {
       printf("[YH_DEBUG][%d][NDC%d] mem_fetch_info for returnq. uid = 0x%x, addr = 0x%x \n", m_gpu->gpu_sim_cycle, id, data->get_request_uid(), data->get_addr());
       m_ndc_cache->display_state(stdout);
   }
   printf("[YH_DEBUG][NDC%d] Print current data\n", id);
   data->print(stdout);
#endif // YH_DEBUG

   // yhyang {
   // when NDC is enabled
   if (!m_config->m_L3_NDC_config.disabled()) {
       //printf("[PSH_DEBUG]ndc_access uid : %d, access_type : %d req_type : %d\n", data->get_request_uid(), data->get_access_type(), data->get_cxl_req_type());
       std::list<cache_event> events;
       unsigned cache_index = (unsigned)-1;

       if ((data->get_cxl_req_type() == CXL_RD_LINE_FILL) || (data->get_cxl_req_type() == CXL_WR_LINE_FILL)) {
           // LINE FILLING request from CXL
           bool fill_success;
           fill_success = ndc_fill_access(cmd);
           if(!fill_success)
               rm_cmd_flag = false;

       } else if (!((data->get_cxl_req_type() == CXL_RD_LINE_FILL) || (data->get_cxl_req_type() == CXL_WR_LINE_FILL))) {
           // no line fill request. On the other word, first request from L2
           // try to access NDC.
           if (m_ndc_cache->data_port_free()) {
               std::list<cache_event> events;
               // enum cache_request_status status = m_L2cache->access(mf->get_addr(), mf, gpu_sim_cycle + gpu_tot_sim_cycle, events);
               bool write_sent = was_write_sent(events);
               bool read_sent = was_read_sent(events);

               if (data->get_request_uid() == TGT_UID) {
                   printf("[YH_DEBUG][uid:%d] Before NDC access\n", data->get_request_uid());
                   data->print(stdout);
                   m_ndc_cache->display_state(stdout);
               }
               // --------------------------------------------------
               // NDC CACHE ACCESS
               // --------------------------------------------------
               cache_request_status status = m_ndc_cache->access(data->get_addr(), data, m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle, events);
               // status is HIT/MISS/RESERVATION_FAIL. Other cases -> assert
               if (data->get_request_uid() == TGT_UID) {
                   printf("[YH_DEBUG][uid:%d] After NDC access..\n", cmd->data->get_request_uid());
                   data->print(stdout);
                   m_ndc_cache->display_state(stdout);
               }

               if (status == HIT) {
                   //printf("[YH_DEBUG][ndc_access==HIT] mf.uid: %d\n", data->get_request_uid());
                   m_memory_partition_unit->ndc_hit_cnt += 1;
                   data->set_ndc_resp(NDC_HIT);
                   data->set_cxl_ret_path(CXL_L2);
                   if (!(data->get_access_type() == L1_WRBK_ACC || data->get_access_type() == L2_WRBK_ACC)) data->set_reply();
                   data->set_status(IN_PARTITION_MC_RETURNQ, m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
                   returnq->push(data);
               } else if (status == MISS) {
                   m_memory_partition_unit->ndc_miss_cnt += 1;
                   // read/write miss cases

                   if (data->get_access_type() == GLOBAL_ACC_R || data->get_access_type() == CONST_ACC_R || data->get_access_type() == TEXTURE_ACC_R || data->get_access_type() == INST_ACC_R ||
                       data->get_access_type() == L1_WR_ALLOC_R || data->get_access_type() == L2_WR_ALLOC_R) {
                       // Read miss. The request will be transfered through NDC miss queue
                       data->set_status(IN_PARTITION_NDC_MISS_QUEUE, m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
                       data->set_ndc_resp(NDC_MISS);
                       data->set_cxl_req_type(CXL_RD_LINE_FILL);
                       data->set_cxl_ret_path(CXL_L2_DRAM);
                       //printf("[PSH_DEBUG][ndc_access==MISS] mf.uid: %d, is_write : %d\n", data->get_request_uid(), data->is_write());
                   
                       // m_return_queue.push(data);
                       data->set_reply();
                   } else if (data->get_access_type() == GLOBAL_ACC_W || data->get_access_type() == L1_WRBK_ACC || data->get_access_type() == L2_WRBK_ACC) {
                       // write miss
                       // WR_MISS original request goes back to returnq for deletion. WR_MISS newly created by NDC is goes to returnq through miss queue with invalid NDC_INVALID resp
                       data->set_ndc_resp(NDC_MISS);
                       data->set_cxl_req_type(CXL_INVALID);
                       data->set_cxl_ret_path(CXL_NONE);
                       //printf("[PSH_DEBUG][ndc_access==MISS] mf.uid: %d, is_write : %d\n", data->get_request_uid(), data->is_write());
                   
                   } else {
                       printf("[YH_DEBUG][NDC%d] Error: invalid access type got NDC MISS. access_type = %d \n", id, data->get_access_type());
                       rm_cmd_flag = false;
                       assert(0);
                   }
               } else {
                   printf("[YH_DEBUG][ndc_access==RESERVATION_FAIL] mf.uid: %d\n", data->get_request_uid());
                   // RESERVATION_FAIL case
                   assert(!write_sent);
                   assert(!read_sent);
                   rm_cmd_flag = false;
                   // debug information
                   if ((id == TGT_MPID) || (TGT_MPID == -1)) {
                       printf("[YH_DEBUG][NDC%d] NDC status is neither HIT nor MISS ????\n", id);
                       printf("[YH_DEBUG][%d][NDC%d] uid = %d, addr = 0x%x, status = %d. Maybe RESERVATION_FAIL. from_rsvq: %d\n", m_gpu->gpu_sim_cycle, id, data->get_request_uid(), data->get_addr(), status,
                              from_rsvq);
                   }
                   // debug information
                   if ((data->get_request_uid() == TGT_UID) || ((id == TGT_MPID) || (TGT_MPID == -1))) {
                       if ((data->get_request_uid() == TGT_UID) /*error_fixed || (err_cnt++ > 10000)*/) {
                           printf("[YH_DEBUG][%d][NDC%d] uid = %d, addr = 0x%x, status = %d. Maybe RESERVATION_FAIL. current rwq len: %d. next data?\n", m_gpu->gpu_sim_cycle, id, data->get_request_uid(),
                                  data->get_addr(), status, rwq->get_length());
                           //YH_DEBUG:printf("[YH_DEBUG] print rwq status. from_rsvq: %d\n", from_rsvq);
                           //YH_DEBUG:{
                           //YH_DEBUG:    fifo_data<dram_req_t> *m_head = NULL;
                           //YH_DEBUG:    // fifo_data<dram_req_t> *m_tail = NULL;
                           //YH_DEBUG:    m_head = rwq->m_head;
                           //YH_DEBUG:    while (m_head) {
                           //YH_DEBUG:        if (m_head->m_data) m_head->m_data->print(stdout);
                           //YH_DEBUG:        // m_tail = m_head;
                           //YH_DEBUG:        m_head = m_head->m_next;
                           //YH_DEBUG:    }
                           //YH_DEBUG:}
                       }
                   }
                   m_memory_partition_unit->ndc_rsv_fail_cnt += 1;
                   // RESERVATION_FAIL
                   // NDC cache lock-up: will try again next cycle
                   // push to ndc_rsv_fail_queue instead of rwq:rwq->push(cmd);   // RESERVATION_FAIL
                   if (!ndc_rsv_fail_queue->full())
                       ndc_rsv_fail_queue->push(cmd);
                   else
                       rwq->push(cmd);
               }
#ifdef YH_DEBUG
               printf("[YH_DEBUG][NDC%d] print status after ndc_cache response treatment\n", id);
               data->print(stdout);
#endif  // YH_DEBUG
           } else {
               m_memory_partition_unit->ndc_data_port_busy_cnt += 1;
               // need to re-push to rsvq or rwq due to data_port is not free
               rm_cmd_flag = false;
               if (from_rsvq)
                   ndc_rsv_fail_queue->push(cmd);
               else
                   rwq->push(cmd);  // data_port not free
           }
       } else {
           printf("[YH_DEBUG][NDC%d] cxl_req_type: %d, waiting_for_fill : %d\n", id, data->get_cxl_req_type(), m_ndc_cache->waiting_for_fill(data));
           printf("[YH_DEBUG][NDC%d] Need to check the wrong status \n", id);
           rm_cmd_flag = false;
           assert(0);
       }
   } else {
       // m_config->m_L3_NDC_config.disabled()
       // original architecture
       data->set_status(IN_PARTITION_MC_RETURNQ, m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
       if (data->get_access_type() != L1_WRBK_ACC && data->get_access_type() != L2_WRBK_ACC) {
           data->set_reply();
           returnq->push(data);
       } else {
           m_memory_partition_unit->set_done(data);
           //printf("[PSH_DEBUG][Check Delete1]\n");
           delete data;
       }
       //printf("[PSH_DEBUG][Check Delete2]\n");
       delete cmd;
       rm_cmd_flag = false;
   }
   // } yhyang

   if (rm_cmd_flag) {
#ifdef YH_DEBUG
      printf("[YH_DEBUG][NDC%d] delete cmd.\n", id);
      if (data)
         printf("[YH_DEBUG][NDC%d] data_uid : %d.\n", id, data->get_request_uid());
#endif // YH_DEBUG
      mem_fetch* check_mf = cmd->data;
      delete cmd;
      //printf("[PSH_DEBUG][Check Delete3] Checking logic uid %d access_type : %d\n", check_mf->get_request_uid(), check_mf->get_access_type());
   }
}

bool ndc_t::ndc_fill_access(dram_req_t *cmd) {
    bool success = true;
    mem_fetch *data = cmd->data;

#ifdef YH_DEBUG
    if ((id == 10) && (m_gpu->gpu_sim_cycle > LOG_ST)) {
        printf("[YH_DEBUG][%d][NDC%d] mem_fetch_info for ndc_fill_queue_is_not_empty, returnq uid = %d, addr = 0x%x \n", m_gpu->gpu_sim_cycle, id, data->get_request_uid(), data->get_addr());
        m_ndc_cache->display_state(stdout);
    }
#endif  // YH_DEBUG

    bool fp_free = m_ndc_cache->fill_port_free();
    bool waiting_fill = m_ndc_cache->waiting_for_fill(data);
    if (!waiting_fill) {
        
#ifdef YH_DEBUG
        printf("[YH_DEBUG][%d][NDC%d] try to fill but there is no waiting_for_fill uid : %d, addr = 0x%x \n", m_gpu->gpu_sim_cycle, id, data->get_request_uid(), data->get_addr());
        m_ndc_cache->display_state(stdout);
#endif  // YH_DEBUG

        data->set_status(IN_PARTITION_NDC_FILL_QUEUE, m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
        data->set_status(IN_PARTITION_MC_RETURNQ, m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
        data->set_ndc_resp(NDC_FILL_DONE);
        returnq->push(data);
        m_memory_partition_unit->ndc_fill_port_busy_cnt += 1;
    } else if (fp_free && waiting_fill) {
#ifdef YH_DEBUG
        printf("[YH_DEBUG][%d][NDC%d] Try to fill from dram_ndc_fill_queue. size: %d\n", m_gpu->gpu_sim_cycle, id, dram_ndc_fill_queue->get_length());
        data->print(stdout);
#endif  // YH_DEBUG

        data->set_status(IN_PARTITION_NDC_FILL_QUEUE, m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
#ifdef YH_DEBUG
        printf("[YH_DEBUG][%d][NDC%d] try to fill uid : %d, addr = 0x%x \n", m_gpu->gpu_sim_cycle, id, data->get_request_uid(), data->get_addr());
#endif  // YH_DEBUG
        m_ndc_cache->fill(data, m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
    } else {
        // fill port is not free
        m_memory_partition_unit->ndc_data_port_busy_cnt += 1;
        // move data from rwq to dram_ndc_fill_queue for waiting fill_port_free
#ifdef YH_DEBUG
        printf("[YH_DEBUG][%d][NDC%d] NDC fill port is not free. move to dram_ndc_fill_queue. size: %d\n", m_gpu->gpu_sim_cycle, id, dram_ndc_fill_queue->get_length());
        data->print(stdout);
#endif  // YH_DEBUG
        if (!dram_ndc_fill_queue->full()) {
            dram_ndc_fill_queue->push(cmd);
        } else {
            printf("[YH_DEBUG][%d][NDC%d] NDC fill queue is full. Please check the queue size and increase it if required\n", m_gpu->gpu_sim_cycle, id);
            printf("[YH_DEBUG][%d] size: %d\n", m_gpu->gpu_sim_cycle, dram_ndc_fill_queue->get_length());
            assert(0);
        }
        //  keep cmd from deletion
        success = false;
    }
    return success;
}

void ndc_t::cycle() {
    // ------------------------------------------------------------
    // NDC cache cycle
    // ------------------------------------------------------------
    m_memory_partition_unit->dram_total_cycle_cnt += 1;
    if (returnq_full()) m_memory_partition_unit->queue_full_cnt_returnq_cache_cycle += 1;
    cache_cycle();  // transfer filled request to returnq, run ndc->cycle() and it fills ndc miss_queue

    // ------------------------------------------------------------
    // reserved fill queue to ndc->fill
    // ------------------------------------------------------------
    if ( returnq_full() && !dram_ndc_fill_queue->empty()) m_memory_partition_unit->queue_full_cnt_returnq_fill_cycle += 1;
    if (!returnq_full() && !dram_ndc_fill_queue->empty()) {
        if (!m_config->m_L3_NDC_config.disabled()) {
            dram_req_t *cmd = dram_ndc_fill_queue->pop();
            //printf("[PSH_DEBUG][ndc_fill_access] uid: %d, ret = %d, req = %d\n", cmd->data->get_request_uid(), cmd->data->get_cxl_req_type(), cmd->data->get_cxl_ret_path());
            
            bool fill_success = ndc_fill_access(cmd);
            //printf("[PSH_DEBUG][ndc_fill_access] uid: %d, ret = %d, req = %d\n", cmd->data->get_request_uid(), cmd->data->get_cxl_req_type(), cmd->data->get_cxl_ret_path());
            if (fill_success) {
                //printf("[PSH_DEBUG][Check Delete4]\n");
                delete cmd;
            }
        }
    }
    // ------------------------------------------------------------------------------------------------------------------------
    // rwq to ndc->access (normal) or ndc_fill (NDC_WR_ALLOC_R or RD_LINE_FILL)
    // ------------------------------------------------------------------------------------------------------------------------
    if (returnq_full()) m_memory_partition_unit->queue_full_cnt_returnq_ndc_cycle += 1;
    if (!returnq_full()) {
        dram_req_t *cmd = rwq->pop();
        // TODO_RWQ:dram_req_t *cmd = rwq->top();
        if (cmd) {
#ifdef DRAM_VIEWCMD
            printf("\tDQ: BK%d Row:%03x Col:%03x", cmd->bk, cmd->row, cmd->col + cmd->dqbytes);
#endif
            cmd->dqbytes += m_config->dram_atom_size;
            if (cmd->dqbytes >= cmd->nbytes) {
                //printf("[PSH_DEBUG][ndc_access] mf.uid: %d\n", cmd->data->get_request_uid());
                ndc_access(cmd, false);
            }
#ifdef DRAM_VIEWCMD
            printf("\n");
#endif
        } else {
            // if rwq is invalid in current cycle, check reservation queue for NDC access......
            if (!ndc_rsv_fail_queue->empty()) {
                dram_req_t *cmd = ndc_rsv_fail_queue->pop();
                ndc_access(cmd, true);
            }
        }
    }

    /* check if the upcoming request is on an idle bank */
    /* Should we modify this so that multiple requests are checked? */

    // YH_DEBUG:printf("[YH_DEBUG] before MC scheduling\n");
    switch (m_config->scheduler_type) {
        case DRAM_FIFO:
            scheduler_fifo();
            break;
        case DRAM_FRFCFS:
            //  yhyang {
            if (!m_config->m_L3_NDC_config.disabled()) {
                scheduler_fffrfcfs();
            } else {
                scheduler_frfcfs();
            }
            break;  // fill request first.
            // } yhyang
        default:
            printf("Error: Unknown DRAM scheduler type\n");
            assert(0);
    }

    // --------------------------------------------------------------------------------
    // // No change for DRAM data transfer for NDC
    // --------------------------------------------------------------------------------
    if (m_config->scheduler_type == DRAM_FRFCFS) {
        unsigned nreqs = m_frfcfs_scheduler->num_pending();
        if (nreqs > max_mrqs) {
            max_mrqs = nreqs;
        }
        ave_mrqs += nreqs;
        ave_mrqs_partial += nreqs;
    } else {
        if (mrqq->get_length() > max_mrqs) {
            max_mrqs = mrqq->get_length();
        }
        ave_mrqs += mrqq->get_length();
        ave_mrqs_partial += mrqq->get_length();
    }

    unsigned k = m_config->nbk;
    bool issued = false;

    // collect row buffer locality, BLP and other statistics
    /////////////////////////////////////////////////////////////////////////
    unsigned int memory_pending = 0;
    for (unsigned i = 0; i < m_config->nbk; i++) {
        if (bk[i]->mrq) memory_pending++;
    }
    banks_1time += memory_pending;
    if (memory_pending > 0) banks_acess_total++;

    unsigned int memory_pending_rw = 0;
    unsigned read_blp_rw = 0;
    unsigned write_blp_rw = 0;
    std::bitset<8> bnkgrp_rw_found;  // assume max we have 8 bank groups

    for (unsigned j = 0; j < m_config->nbk; j++) {
        unsigned grp = get_bankgrp_number(j);
        if (bk[j]->mrq &&
            (((bk[j]->curr_row == bk[j]->mrq->row) && (bk[j]->mrq->rw == READ) &&
            (bk[j]->state == BANK_ACTIVE)))) {
        memory_pending_rw++;
        read_blp_rw++;
        bnkgrp_rw_found.set(grp);
        } else if (bk[j]->mrq &&
                (((bk[j]->curr_row == bk[j]->mrq->row) &&
                    (bk[j]->mrq->rw == WRITE) && (bk[j]->state == BANK_ACTIVE)))) {
        memory_pending_rw++;
        write_blp_rw++;
        bnkgrp_rw_found.set(grp);
        }
    }
    banks_time_rw += memory_pending_rw;
    bkgrp_parallsim_rw += bnkgrp_rw_found.count();
    if (memory_pending_rw > 0) {
        write_to_read_ratio_blp_rw_average +=
            (double)write_blp_rw / (write_blp_rw + read_blp_rw);
        banks_access_rw_total++;
    }

    unsigned int memory_Pending_ready = 0;
    for (unsigned j = 0; j < m_config->nbk; j++) {
        unsigned grp = get_bankgrp_number(j);
        if (bk[j]->mrq &&
            ((!CCDc && !bk[j]->RCDc && !(bkgrp[grp]->CCDLc) &&
            (bk[j]->curr_row == bk[j]->mrq->row) && (bk[j]->mrq->rw == READ) &&
            (WTRc == 0) && (bk[j]->state == BANK_ACTIVE) && !rwq->full()) ||
            (!CCDc && !bk[j]->RCDWRc && !(bkgrp[grp]->CCDLc) &&
            (bk[j]->curr_row == bk[j]->mrq->row) && (bk[j]->mrq->rw == WRITE) &&
            (RTWc == 0) && (bk[j]->state == BANK_ACTIVE) && !rwq->full()))) {
        memory_Pending_ready++;
        }
    }
    banks_time_ready += memory_Pending_ready;
    if (memory_Pending_ready > 0) banks_access_ready_total++;
    ///////////////////////////////////////////////////////////////////////////////////

    bool issued_col_cmd = false;
    bool issued_row_cmd = false;

    if (m_config->dual_bus_interface) {
        // dual bus interface
        // issue one row command and one column command
        for (unsigned i = 0; i < m_config->nbk; i++) {
        unsigned j = (i + prio) % m_config->nbk;
        issued_col_cmd = issue_col_command(j);
        if (issued_col_cmd) break;
        }
        for (unsigned i = 0; i < m_config->nbk; i++) {
        unsigned j = (i + prio) % m_config->nbk;
        issued_row_cmd = issue_row_command(j);
        if (issued_row_cmd) break;
        }
        for (unsigned i = 0; i < m_config->nbk; i++) {
        unsigned j = (i + prio) % m_config->nbk;
        if (!bk[j]->mrq) {
            if (!CCDc && !RRDc && !RTWc && !WTRc && !bk[j]->RCDc && !bk[j]->RASc &&
                !bk[j]->RCc && !bk[j]->RPc && !bk[j]->RCDWRc)
            k--;
            bk[j]->n_idle++;
        }
        }
    } else {
        // single bus interface
        // issue only one row/column command
        for (unsigned i = 0; i < m_config->nbk; i++) {
        unsigned j = (i + prio) % m_config->nbk;
        if (!issued_col_cmd) issued_col_cmd = issue_col_command(j);

        if (!issued_col_cmd && !issued_row_cmd)
            issued_row_cmd = issue_row_command(j);

        if (!bk[j]->mrq) {
            if (!CCDc && !RRDc && !RTWc && !WTRc && !bk[j]->RCDc && !bk[j]->RASc &&
                !bk[j]->RCc && !bk[j]->RPc && !bk[j]->RCDWRc)
            k--;
            bk[j]->n_idle++;
        }
        }
    }

    issued = issued_row_cmd || issued_col_cmd;
    if (!issued) {
        n_nop++;
        n_nop_partial++;
    #ifdef DRAM_VIEWCMD
        printf("\tNOP                        ");
    #endif
    }
    if (k) {
        n_activity++;
        n_activity_partial++;
    }
    n_cmd++;
    n_cmd_partial++;
    if (issued) {
        issued_total++;
        if (issued_col_cmd && issued_row_cmd) issued_two++;
    }
    if (issued_col_cmd) issued_total_col++;
    if (issued_row_cmd) issued_total_row++;

    // Collect some statistics
    // check the limitation, see where BW is wasted?
    /////////////////////////////////////////////////////////
    unsigned int memory_pending_found = 0;
    for (unsigned i = 0; i < m_config->nbk; i++) {
        if (bk[i]->mrq) memory_pending_found++;
    }
    if (memory_pending_found > 0) banks_acess_total_after++;

    bool memory_pending_rw_found = false;
    for (unsigned j = 0; j < m_config->nbk; j++) {
        if (bk[j]->mrq &&
            (((bk[j]->curr_row == bk[j]->mrq->row) && (bk[j]->mrq->rw == READ) &&
            (bk[j]->state == BANK_ACTIVE)) ||
            ((bk[j]->curr_row == bk[j]->mrq->row) && (bk[j]->mrq->rw == WRITE) &&
            (bk[j]->state == BANK_ACTIVE))))
        memory_pending_rw_found = true;
    }

    if (issued_col_cmd || CCDc)
        util_bw++;
    else if (memory_pending_rw_found) {
        wasted_bw_col++;
        for (unsigned j = 0; j < m_config->nbk; j++) {
        unsigned grp = get_bankgrp_number(j);
        // read
        if (bk[j]->mrq &&
            (((bk[j]->curr_row == bk[j]->mrq->row) && (bk[j]->mrq->rw == READ) &&
                (bk[j]->state == BANK_ACTIVE)))) {
            if (bk[j]->RCDc) RCDc_limit++;
            if (bkgrp[grp]->CCDLc) CCDLc_limit++;
            if (WTRc) WTRc_limit++;
            if (CCDc) CCDc_limit++;
            if (rwq->full()) rwq_limit++;
            if (bkgrp[grp]->CCDLc && !WTRc) CCDLc_limit_alone++;
            if (!bkgrp[grp]->CCDLc && WTRc) WTRc_limit_alone++;
        }
        // write
        else if (bk[j]->mrq &&
                ((bk[j]->curr_row == bk[j]->mrq->row) &&
                    (bk[j]->mrq->rw == WRITE) && (bk[j]->state == BANK_ACTIVE))) {
            if (bk[j]->RCDWRc) RCDWRc_limit++;
            if (bkgrp[grp]->CCDLc) CCDLc_limit++;
            if (RTWc) RTWc_limit++;
            if (CCDc) CCDc_limit++;
            if (rwq->full()) rwq_limit++;
            if (bkgrp[grp]->CCDLc && !RTWc) CCDLc_limit_alone++;
            if (!bkgrp[grp]->CCDLc && RTWc) RTWc_limit_alone++;
        }
        }
    } else if (memory_pending_found)
        wasted_bw_row++;
    else if (!memory_pending_found)
        idle_bw++;
    else
        assert(1);

    /////////////////////////////////////////////////////////

   // decrements counters once for each time dram_issueCMD is called
   DEC2ZERO(RRDc);
   DEC2ZERO(CCDc);
   DEC2ZERO(RTWc);
   DEC2ZERO(WTRc);
   for (unsigned j=0;j<m_config->nbk;j++) {
      DEC2ZERO(bk[j]->RCDc);
      DEC2ZERO(bk[j]->RASc);
      DEC2ZERO(bk[j]->RCc);
      DEC2ZERO(bk[j]->RPc);
      DEC2ZERO(bk[j]->RCDWRc);
      DEC2ZERO(bk[j]->WTPc);
      DEC2ZERO(bk[j]->RTPc);
   }
   for (unsigned j=0; j<m_config->nbkgrp; j++) {
	   DEC2ZERO(bkgrp[j]->CCDLc);
	   DEC2ZERO(bkgrp[j]->RTPLc);
   }

#ifdef DRAM_VISUALIZE
   visualize();
#endif
}

// yhyang {
void ndc_t::push_from_cxl(class mem_fetch *data) {


    //printf("[PSH_DEBUG][dram_t::push_from_cxl] u_id: 0x%x, chipid: 0x%x\n", data->get_request_uid(), data->get_tlx_addr().chip);
                                      
    assert(id == data->get_tlx_addr().chip);  // Ensure request is in correct memory partition

    dram_req_t *mrq = new dram_req_t(data, m_config->nbk, m_config->dram_bnk_indexing_policy,
                     m_memory_partition_unit->get_mgpu());
    if (data->get_cxl_req_type() == CXL_RD_LINE_FILL) {
        mrq->rw = WRITE;  // change RD_LINE_FILL request's DRAM_REQ type to WRITE
    }
    data->set_status(IN_PARTITION_MC_INTERFACE_QUEUE, m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
    fill_mrqq->push(mrq);

    // stats...
    n_req += 1;
    n_req_partial += 1;
    if (m_config->scheduler_type == DRAM_FRFCFS) {
        unsigned nreqs = m_frfcfs_scheduler->num_pending();
        if (nreqs > max_mrqs_temp) max_mrqs_temp = nreqs;
    } else {
        max_mrqs_temp = (max_mrqs_temp > mrqq->get_length()) ? max_mrqs_temp : mrqq->get_length();
    }
    //m_stats->memlatstat_dram_access(data);
}
bool ndc_t::full_from_cxl() const {
    //printf("[PSH_DEBUG][mp%d][full_from_cxl] error check\n");
    if (m_config->scheduler_type == DRAM_FRFCFS) {
        //printf("[PSH_DEBUG][mp%d][full_from_cxl] error check\n");
        if (m_config->gpgpu_frfcfs_dram_sched_queue_size == 0) return false;
        return m_frfcfs_scheduler->num_pending() >= m_config->gpgpu_frfcfs_dram_sched_queue_size;
    } else
        return fill_mrqq->full();
}
void ndc_t::scheduler_fffrfcfs() {
    unsigned mrq_latency;
    frfcfs_scheduler *sched = m_frfcfs_scheduler;
    // fill_mrqq first
    while (!fill_mrqq->empty() && (!m_config->gpgpu_frfcfs_dram_sched_queue_size || sched->num_pending() < m_config->gpgpu_frfcfs_dram_sched_queue_size)) {
        dram_req_t *req = fill_mrqq->pop();
#ifdef YH_DEBUG
        printf("[YH_DEBUG][mp%d][scheduler_fffrfcfs] fill_mrqq is not empty. req->data->uid : %d\n", id, req->data->get_request_uid());
#endif  // YH_DEBUG

        // Power stats
        // if(req->data->get_type() != READ_REPLY && req->data->get_type() != WRITE_ACK)
        m_stats->total_n_access++;

        if (req->data->get_type() == WRITE_REQUEST) {
            m_stats->total_n_writes++;
        } else if (req->data->get_type() == READ_REQUEST) {
            m_stats->total_n_reads++;
        }

        req->data->set_status(IN_PARTITION_MC_INPUT_QUEUE, m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
        sched->add_req(req);
        //printf("[PSH_DEBUG]fill_mrqq->pop and add to sched : uid %d\n", req->data->get_request_uid());
    }
    while (!mrqq->empty() && (!m_config->gpgpu_frfcfs_dram_sched_queue_size || sched->num_pending() < m_config->gpgpu_frfcfs_dram_sched_queue_size)) {
        dram_req_t *req = mrqq->pop();
#ifdef YH_DEBUG
        printf("[YH_DEBUG][mp%d][scheduler_fffrfcfs] mrqq is not empty. req->data->uid : %d\n", id, req->data->get_request_uid());
#endif  // YH_DEBUG

        // Power stats
        // if(req->data->get_type() != READ_REPLY && req->data->get_type() != WRITE_ACK)
        m_stats->total_n_access++;

        if (req->data->get_type() == WRITE_REQUEST) {
            m_stats->total_n_writes++;
        } else if (req->data->get_type() == READ_REQUEST) {
            m_stats->total_n_reads++;
        }

        req->data->set_status(IN_PARTITION_MC_INPUT_QUEUE, m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
        sched->add_req(req);
    }

    dram_req_t *req;
    unsigned i;
    for (i = 0; i < m_config->nbk; i++) {
        unsigned b = (i + prio) % m_config->nbk;
        if (!bk[b]->mrq) {
            req = sched->schedule(b, bk[b]->curr_row);
            if (req) {
                //if(req->data->get_access_type() == NDC_LINEFILL_W)
                    //if(id==0) printf("[YH_DEBUG][mp%d][scheduler_fffrfcfs] bank scheduling. bank: %d, uid: %d access_t %d\n", id, b, req->data->get_request_uid(), req->data->get_access_type());
                req->data->set_status(IN_PARTITION_MC_BANK_ARB_QUEUE, m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
                prio = (prio + 1) % m_config->nbk;
                bk[b]->mrq = req;
                if (m_config->gpgpu_memlatency_stat) {
                    mrq_latency = m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle - bk[b]->mrq->timestamp;
                    bk[b]->mrq->timestamp = m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle;
                    m_stats->mrq_lat_table[LOGB2(mrq_latency)]++;
                    if (mrq_latency > m_stats->max_mrq_latency) {
                        m_stats->max_mrq_latency = mrq_latency;
                    }
                }

                break;
            }
        }
    }
}

void ndc_t::accumulate_L3_NDCcache_stats(class cache_stats &l3_ndc_stats) const {
    if (!m_config->m_L3_NDC_config.disabled()) {
        l3_ndc_stats += m_ndc_cache->get_stats();
    }
}

void ndc_t::get_L3_NDCcache_sub_stats(struct cache_sub_stats &css) const {
    if (!m_config->m_L3_NDC_config.disabled()) {
        m_ndc_cache->get_sub_stats(css);
    }
}

// } yhyang