// Copyright (c) 2009-2011, Tor M. Aamodt, Ali Bakhoda, George L. Yuan,
// The University of British Columbia
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution. Neither the name of
// The University of British Columbia nor the names of its contributors may be
// used to endorse or promote products derived from this software without
// specific prior written permission.
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

#include "dram_sched.h"
#include "../abstract_hardware_model.h"
#include "gpu-misc.h"
#include "gpu-sim.h"
#include "mem_latency_stat.h"

frfcfs_scheduler::frfcfs_scheduler(const memory_config *config, dram_t *dm,
                                   memory_stats_t *stats) {
  m_config = config;
  m_stats = stats;
  m_num_pending = 0;
  m_num_write_pending = 0;
  m_dram = dm;
  m_queue = new std::list<dram_req_t *>[m_config->nbk];
  m_bins = new std::map<
      unsigned, std::list<std::list<dram_req_t *>::iterator> >[m_config->nbk];
  m_last_row =
      new std::list<std::list<dram_req_t *>::iterator> *[m_config->nbk];
  curr_row_service_time = new unsigned[m_config->nbk];
  row_service_timestamp = new unsigned[m_config->nbk];
  for (unsigned i = 0; i < m_config->nbk; i++) {
    m_queue[i].clear();
    m_bins[i].clear();
    m_last_row[i] = NULL;
    curr_row_service_time[i] = 0;
    row_service_timestamp[i] = 0;
  }
  if (m_config->seperate_write_queue_enabled) {
    m_write_queue = new std::list<dram_req_t *>[m_config->nbk];
    m_write_bins = new std::map<
        unsigned, std::list<std::list<dram_req_t *>::iterator> >[m_config->nbk];
    m_last_write_row =
        new std::list<std::list<dram_req_t *>::iterator> *[m_config->nbk];

    for (unsigned i = 0; i < m_config->nbk; i++) {
      m_write_queue[i].clear();
      m_write_bins[i].clear();
      m_last_write_row[i] = NULL;
    }
  }
  m_mode = READ_MODE;
}

void frfcfs_scheduler::add_req(dram_req_t *req) {
  if (m_config->seperate_write_queue_enabled && req->data->is_write()) {
    assert(m_num_write_pending < m_config->gpgpu_frfcfs_dram_write_queue_size);
    m_num_write_pending++;
    m_write_queue[req->bk].push_front(req);
    std::list<dram_req_t *>::iterator ptr = m_write_queue[req->bk].begin();
    m_write_bins[req->bk][req->row].push_front(ptr);  // newest reqs to the
                                                      // front
  } else {
    assert(m_num_pending < m_config->gpgpu_frfcfs_dram_sched_queue_size);
    m_num_pending++;
    m_queue[req->bk].push_front(req);
    std::list<dram_req_t *>::iterator ptr = m_queue[req->bk].begin();
    m_bins[req->bk][req->row].push_front(ptr);  // newest reqs to the front
  }
}

void frfcfs_scheduler::data_collection(unsigned int bank) {
  if (m_dram->m_gpu->gpu_sim_cycle > row_service_timestamp[bank]) {
    curr_row_service_time[bank] =
        m_dram->m_gpu->gpu_sim_cycle - row_service_timestamp[bank];
    if (curr_row_service_time[bank] >
        m_stats->max_servicetime2samerow[m_dram->id][bank])
      m_stats->max_servicetime2samerow[m_dram->id][bank] =
          curr_row_service_time[bank];
  }
  curr_row_service_time[bank] = 0;
  row_service_timestamp[bank] = m_dram->m_gpu->gpu_sim_cycle;
  if (m_stats->concurrent_row_access[m_dram->id][bank] >
      m_stats->max_conc_access2samerow[m_dram->id][bank]) {
    m_stats->max_conc_access2samerow[m_dram->id][bank] =
        m_stats->concurrent_row_access[m_dram->id][bank];
  }
  m_stats->concurrent_row_access[m_dram->id][bank] = 0;
  m_stats->num_activates[m_dram->id][bank]++;
}

dram_req_t *frfcfs_scheduler::schedule(unsigned bank, unsigned curr_row) {
  // row
  bool rowhit = true;
  std::list<dram_req_t *> *m_current_queue = m_queue;
  std::map<unsigned, std::list<std::list<dram_req_t *>::iterator> >
      *m_current_bins = m_bins;
  std::list<std::list<dram_req_t *>::iterator> **m_current_last_row =
      m_last_row;

  if (m_config->seperate_write_queue_enabled) {
    if (m_mode == READ_MODE &&
        ((m_num_write_pending >= m_config->write_high_watermark)
         // || (m_queue[bank].empty() && !m_write_queue[bank].empty())
         )) {
      m_mode = WRITE_MODE;
    } else if (m_mode == WRITE_MODE &&
               ((m_num_write_pending < m_config->write_low_watermark)
                //  || (!m_queue[bank].empty() && m_write_queue[bank].empty())
                )) {
      m_mode = READ_MODE;
    }
  }

  if (m_mode == WRITE_MODE) {
    m_current_queue = m_write_queue;
    m_current_bins = m_write_bins;
    m_current_last_row = m_last_write_row;
  }

  if (m_current_last_row[bank] == NULL) {
    if (m_current_queue[bank].empty()) return NULL;

    std::map<unsigned, std::list<std::list<dram_req_t *>::iterator> >::iterator
        bin_ptr = m_current_bins[bank].find(curr_row);
    if (bin_ptr == m_current_bins[bank].end()) {
      dram_req_t *req = m_current_queue[bank].back();
      bin_ptr = m_current_bins[bank].find(req->row);
      assert(bin_ptr !=
             m_current_bins[bank].end());  // where did the request go???
      m_current_last_row[bank] = &(bin_ptr->second);
      data_collection(bank);
      rowhit = false;
    } else {
      m_current_last_row[bank] = &(bin_ptr->second);
      rowhit = true;
    }
  }
  std::list<dram_req_t *>::iterator next = m_current_last_row[bank]->back();
  dram_req_t *req = (*next);

  // rowblp stats
  m_dram->access_num++;
  bool is_write = req->data->is_write();
  if (is_write)
    m_dram->write_num++;
  else
    m_dram->read_num++;

  if (rowhit) {
    m_dram->hits_num++;
    if (is_write)
      m_dram->hits_write_num++;
    else
      m_dram->hits_read_num++;
  }

  m_stats->concurrent_row_access[m_dram->id][bank]++;
  m_stats->row_access[m_dram->id][bank]++;
  m_current_last_row[bank]->pop_back();

  m_current_queue[bank].erase(next);
  if (m_current_last_row[bank]->empty()) {
    m_current_bins[bank].erase(req->row);
    m_current_last_row[bank] = NULL;
  }
#ifdef DEBUG_FAST_IDEAL_SCHED
  if (req)
    printf("%08u : DRAM(%u) scheduling memory request to bank=%u, row=%u\n",
           (unsigned)gpu_sim_cycle, m_dram->id, req->bk, req->row);
#endif

  if (m_config->seperate_write_queue_enabled && req->data->is_write()) {
    assert(req != NULL && m_num_write_pending != 0);
    m_num_write_pending--;
  } else {
    assert(req != NULL && m_num_pending != 0);
    m_num_pending--;
  }

  return req;
}

void frfcfs_scheduler::print(FILE *fp) {
  for (unsigned b = 0; b < m_config->nbk; b++) {
    printf(" %u: queue length = %u\n", b, (unsigned)m_queue[b].size());
  }
}

void dram_t::scheduler_frfcfs() {
  unsigned mrq_latency;
  frfcfs_scheduler *sched = m_frfcfs_scheduler;
  while (!mrqq->empty()) {
    dram_req_t *req = mrqq->pop();

    // Power stats
    // if(req->data->get_type() != READ_REPLY && req->data->get_type() !=
    // WRITE_ACK)
    m_stats->total_n_access++;

    if (req->data->get_type() == WRITE_REQUEST) {
      m_stats->total_n_writes++;
    } else if (req->data->get_type() == READ_REQUEST) {
      m_stats->total_n_reads++;
    }

    req->data->set_status(IN_PARTITION_MC_INPUT_QUEUE,
                          m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
    sched->add_req(req);
  }

  dram_req_t *req;
  unsigned i;
  for (i = 0; i < m_config->nbk; i++) {
    unsigned b = (i + prio) % m_config->nbk;
    if (!bk[b]->mrq) {
      req = sched->schedule(b, bk[b]->curr_row);

      if (req) {
        req->data->set_status(IN_PARTITION_MC_BANK_ARB_QUEUE,
                              m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
        prio = (prio + 1) % m_config->nbk;
        bk[b]->mrq = req;
        if (m_config->gpgpu_memlatency_stat) {
          mrq_latency = m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle -
                        bk[b]->mrq->timestamp;
          m_stats->tot_mrq_latency += mrq_latency;
          m_stats->tot_mrq_num++;
          bk[b]->mrq->timestamp =
              m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle;
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

//pshyun_reservation_fail_debug
dram_req_t *fffrfcfs_scheduler::schedule_fill_only(unsigned bank, unsigned curr_row) {
    // row
    bool rowhit = true;
    std::list<dram_req_t *> *m_current_queue = m_queue;
    std::map<unsigned, std::list<std::list<dram_req_t *>::iterator>> *m_current_bins = m_bins;
    std::list<std::list<dram_req_t *>::iterator> **m_current_last_row = m_last_row;

    if (m_config->seperate_write_queue_enabled) {
        if (m_mode == READ_MODE && ((m_num_write_pending >= m_config->write_high_watermark)
                                    // || (m_queue[bank].empty() && !m_write_queue[bank].empty())
                                    )) {
            m_mode = WRITE_MODE;
        } else if (m_mode == WRITE_MODE && ((m_num_write_pending < m_config->write_low_watermark)
                                            //  || (!m_queue[bank].empty() && m_write_queue[bank].empty())
                                            )) {
            m_mode = READ_MODE;
        }
    }

    if (m_mode == WRITE_MODE) {
        m_current_queue = m_write_queue;
        m_current_bins = m_write_bins;
        m_current_last_row = m_last_write_row;
    }

    // yhyang check fill request existence
    bool has_fill_req = false;
    for (std::list<dram_req_t *>::iterator it = m_current_queue[bank].begin(); it != m_current_queue[bank].end(); ++it) {
        dram_req_t *req = *it;
        if (req->data->is_cxl_fill_req()) {
            has_fill_req = true;
            break;
        }
    }
    if (!has_fill_req) return NULL;

    if (true /*m_current_last_row[bank] == NULL  //fffrfcfs always check last row is having fill_req//*/) {
        if (m_current_queue[bank].empty()) return NULL;

        std::map<unsigned, std::list<std::list<dram_req_t *>::iterator>>::iterator bin_ptr = m_current_bins[bank].find(curr_row);

        if (bin_ptr == m_current_bins[bank].end()) {
            rowhit = false;
        } else {
            m_current_last_row[bank] = &(bin_ptr->second);
            // check fill request existence in current opened row. If not, it is not rowhit.
            // yhyang check fill request existence in current row
            bool has_fill_req = false;
            // TODO:for (std::list<dram_req_t *>::iterator it = m_current_last_row[bank]->begin();
            // TODO:		it != m_current_last_row[bank]->end(); ++it) {
            // TODO:	dram_req_t *req = *it;
            // TODO:	if (req->data->is_cxl_fill_req()) {
            // TODO:		has_fill_req = true;
            // TODO:		break;
            // TODO:	}
            // TODO:}
            for (std::list<std::list<dram_req_t *>::iterator>::iterator it = m_current_last_row[bank]->begin(); it != m_current_last_row[bank]->end(); ++it) {
                dram_req_t *req = **it;  // *it: iterator to dram_req_t*, so **it gives dram_req_t*
                if (req->data->is_cxl_fill_req()) {
                    has_fill_req = true;
                    break;
                }
            }

            if (has_fill_req)
                rowhit = true;
            else
                rowhit = false;
        }

        if (!rowhit) {
            // TODO:dram_req_t *req = m_current_queue[bank].back();
            // TODO:bin_ptr = m_current_bins[bank].find(req->row);
            // TODO:assert(bin_ptr !=
            // TODO:		m_current_bins[bank].end());  // where did the request go???

            // Reverse iterate to find a valid cxl_fill_req
            std::list<dram_req_t *>::reverse_iterator rit;
            dram_req_t *selected = NULL;

            for (rit = m_current_queue[bank].rbegin(); rit != m_current_queue[bank].rend(); ++rit) {
                dram_req_t *candidate = *rit;
                if (candidate->data->is_cxl_fill_req()) {
                    selected = candidate;
                    break;
                }
            }
            bin_ptr = m_current_bins[bank].find(selected->row);
            assert(bin_ptr != m_current_bins[bank].end());  // request must exist in bins

            m_current_last_row[bank] = &(bin_ptr->second);
            data_collection(bank);
            // rowhit = false;
        } else {
            m_current_last_row[bank] = &(bin_ptr->second);
            // rowhit = true;
        }
    }
    // TODO:std::list<dram_req_t *>::iterator next = m_current_last_row[bank]->back();
    // TODO:dram_req_t *req = (*next);
    // TODO::dram_req_t *req = NULL;
    // TODO::std::list<dram_req_t *>::iterator next;
    // TODO::// find oldest fill request in the row
    // TODO::for (auto rit = m_current_last_row[bank]->rbegin(); rit != m_current_last_row[bank]->rend(); ++rit) {
    // TODO::	if ((*rit)->data->is_cxl_fill_req()) {
    // TODO::		next = std::prev(rit.base()); // reverse_iterator → iterator 변환
    // TODO::		req = *next;
    // TODO::		break;
    // TODO::	}
    // TODO::}

    dram_req_t *req = NULL;
    std::list<dram_req_t *>::iterator next;
    // find oldest fill request in the row
    for (auto rit = m_current_last_row[bank]->rbegin(); rit != m_current_last_row[bank]->rend(); ++rit) {
        dram_req_t *candidate = *(*rit);  // *rit: iterator, *(*rit): dram_req_t*
        if (candidate->data->is_cxl_fill_req()) {
            next = *rit;  // *rit 자체가 std::list<dram_req_t*>::iterator
            req = candidate;
            break;
        }
    }
#ifdef YH_DEBUG
    if (req == NULL) {
        printf("[YH_DEBUG][mp%d][bank%d]][%d] finding fill_request was failed\n", m_dram->id, bank, m_dram->m_gpu->gpu_sim_cycle);
        printf("[YH_DEBUG][mp%d][bank%d]][%d] next: %d, req: %d\n", m_dram->id, bank, m_dram->m_gpu->gpu_sim_cycle, next, req);
        if (m_current_last_row[bank] != NULL) {
            printf("[YH_DEBUG] m_current_last_row[%d] contents:\n", bank);
            for (std::list<std::list<dram_req_t *>::iterator>::iterator it = m_current_last_row[bank]->begin(); it != m_current_last_row[bank]->end(); ++it) {
                dram_req_t *r = **it;  // iterator가 가리키는 list의 요소가 dram_req_t*
                printf("  req @%p | row=%u bank=%u is_fill=%d is_write=%d\n", r, r->row, r->bk, r->data->is_cxl_fill_req(), r->data->is_write());
            }
        } else {
            printf("[YH_DEBUG] m_current_last_row[%d] is NULL\n", bank);
        }

        assert(0);  // why cannot find fill request?
    }
#endif // YH_DEBUG

    // find iterator exactly pointing the req in m_current_last_row[bank]
    std::list<std::list<dram_req_t *>::iterator>::iterator it_in_last_row;
    for (it_in_last_row = m_current_last_row[bank]->begin(); it_in_last_row != m_current_last_row[bank]->end(); ++it_in_last_row) {
        if (*it_in_last_row == next) break;
    }
    assert(it_in_last_row != m_current_last_row[bank]->end());

    // rowblp stats
    m_dram->access_num++;
    bool is_write = req->data->is_write();
    if (is_write)
        m_dram->write_num++;
    else
        m_dram->read_num++;

    if (rowhit) {
        m_dram->hits_num++;
        if (is_write)
            m_dram->hits_write_num++;
        else
            m_dram->hits_read_num++;
    }

    m_stats->concurrent_row_access[m_dram->id][bank]++;
    m_stats->row_access[m_dram->id][bank]++;
    // yhyang:m_current_last_row[bank]->pop_back();
    m_current_last_row[bank]->erase(it_in_last_row);

    m_current_queue[bank].erase(next);
    if (m_current_last_row[bank]->empty()) {
        m_current_bins[bank].erase(req->row);
        m_current_last_row[bank] = NULL;
    }
#ifdef DEBUG_FAST_IDEAL_SCHED
    if (req) printf("%08u : DRAM(%u) scheduling memory request to bank=%u, row=%u\n", (unsigned)gpu_sim_cycle, m_dram->id, req->bk, req->row);
#endif

    if (m_config->seperate_write_queue_enabled && req->data->is_write()) {
        assert(req != NULL && m_num_write_pending != 0);
        m_num_write_pending--;
    } else {
        assert(req != NULL && m_num_pending != 0);
        m_num_pending--;
    }

    return req;
}

dram_req_t *fffrfcfs_scheduler::schedule(unsigned bank, unsigned curr_row) {
  // row
  bool rowhit = true;
  std::list<dram_req_t *> *m_current_queue = m_queue;
  std::map<unsigned, std::list<std::list<dram_req_t *>::iterator> >
      *m_current_bins = m_bins;
  std::list<std::list<dram_req_t *>::iterator> **m_current_last_row =
      m_last_row;

  if (m_config->seperate_write_queue_enabled) {
    if (m_mode == READ_MODE &&
        ((m_num_write_pending >= m_config->write_high_watermark)
         // || (m_queue[bank].empty() && !m_write_queue[bank].empty())
         )) {
      m_mode = WRITE_MODE;
    } else if (m_mode == WRITE_MODE &&
               ((m_num_write_pending < m_config->write_low_watermark)
                //  || (!m_queue[bank].empty() && m_write_queue[bank].empty())
                )) {
      m_mode = READ_MODE;
    }
  }

  if (m_mode == WRITE_MODE) {
    m_current_queue = m_write_queue;
    m_current_bins = m_write_bins;
    m_current_last_row = m_last_write_row;
  }

  if (m_current_last_row[bank] == NULL) {
    if (m_current_queue[bank].empty()) return NULL;

    std::map<unsigned, std::list<std::list<dram_req_t *>::iterator> >::iterator
        bin_ptr = m_current_bins[bank].find(curr_row);
    if (bin_ptr == m_current_bins[bank].end()) {
      dram_req_t *req = m_current_queue[bank].back();
      bin_ptr = m_current_bins[bank].find(req->row);
      assert(bin_ptr !=
             m_current_bins[bank].end());  // where did the request go???
      m_current_last_row[bank] = &(bin_ptr->second);
      data_collection(bank);
      rowhit = false;
    } else {
      m_current_last_row[bank] = &(bin_ptr->second);
      rowhit = true;
    }
  }
  std::list<dram_req_t *>::iterator next = m_current_last_row[bank]->back();
  dram_req_t *req = (*next);

  // rowblp stats
  m_dram->access_num++;
  bool is_write = req->data->is_write();
  if (is_write)
    m_dram->write_num++;
  else
    m_dram->read_num++;

  if (rowhit) {
    m_dram->hits_num++;
    if (is_write)
      m_dram->hits_write_num++;
    else
      m_dram->hits_read_num++;
  }

  m_stats->concurrent_row_access[m_dram->id][bank]++;
  m_stats->row_access[m_dram->id][bank]++;
  m_current_last_row[bank]->pop_back();

  m_current_queue[bank].erase(next);
  if (m_current_last_row[bank]->empty()) {
    m_current_bins[bank].erase(req->row);
    m_current_last_row[bank] = NULL;
  }
#ifdef DEBUG_FAST_IDEAL_SCHED
  if (req)
    printf("%08u : DRAM(%u) scheduling memory request to bank=%u, row=%u\n",
           (unsigned)gpu_sim_cycle, m_dram->id, req->bk, req->row);
#endif

  if (m_config->seperate_write_queue_enabled && req->data->is_write()) {
    assert(req != NULL && m_num_write_pending != 0);
    m_num_write_pending--;
  } else {
    assert(req != NULL && m_num_pending != 0);
    m_num_pending--;
  }

  return req;
}

void ndc_t::scheduler_fffrfcfs() {
    unsigned mrq_latency;
    //yhyang 250611_15 for scheduling only ndc_fill when rsv_fail_queue is full:frfcfs_scheduler *sched = m_frfcfs_scheduler;
    fffrfcfs_scheduler *sched = m_frfcfs_scheduler;
    // fill_mrqq first
    while (!fill_mrqq->empty() && (!m_config->gpgpu_frfcfs_dram_sched_queue_size || (sched->num_pending()+0) < m_config->gpgpu_frfcfs_dram_sched_queue_size)) {
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
    while (!mrqq->empty() && (!m_config->gpgpu_frfcfs_dram_sched_queue_size || (sched->num_pending()+1) < m_config->gpgpu_frfcfs_dram_sched_queue_size)) {
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
			//yhyang 250611_15 for scheduling only ndc_fill when rsv_fail_queue is full:req = sched->schedule(b, bk[b]->curr_row);
      assert(ndc_rsv_fail_queue->get_max_len() > m_config->nbk);  // keep rsv_fail_queue size is greater than number of bank for protecting deadlock
			if(ndc_rsv_fail_queue->get_length() >= (ndc_rsv_fail_queue->get_max_len() - m_config->nbk)) {	// each bk[b]->mrq is merged to rwq. so before rsv_fail_queue full, number of request (same with number of bank) might be scheduled...
				req = sched->schedule_fill_only(b, bk[b]->curr_row );
			}
			else {
				req = sched->schedule(b, bk[b]->curr_row);
			}
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
void print_queue(std::list<dram_req_t *> *queue, const std::string &label = "m_queue") {
    printf("===== %s =====\n", label.c_str());
    for (auto it = queue->begin(); it != queue->end(); ++it) {
        dram_req_t *req = *it;
        if (req && req->data) {
            printf("  [REQ] ");
            req->data->print(stdout);  // or req->data->print(fp) if using a FILE*
        } else {
            printf("  [REQ] NULL or missing data\n");
        }
    }
}

void print_bins(std::map<unsigned, std::list<std::list<dram_req_t*>::iterator>> *bins,
                std::list<dram_req_t *> *queue,
                const std::string &label = "m_bins") {
    printf("===== %s =====\n", label.c_str());
    for (auto &entry : *bins) {
        unsigned row = entry.first;
        printf("Row %u:\n", row);
        for (auto it : entry.second) {
            if (it != queue->end()) {
                dram_req_t *req = *it;
                if (req && req->data) {
                    printf("  -> ");
                    req->data->print(stdout);
                } else {
                    printf("  -> NULL or missing data\n");
                }
            } else {
                printf("  -> Invalid iterator (end)\n");
            }
        }
    }
}

void print_last_row(std::list<std::list<dram_req_t*>::iterator> **last_row,
                    int num_banks,
                    std::list<dram_req_t *> *queue,
                    const std::string &label = "m_last_row") {
    printf("===== %s =====\n", label.c_str());
    for (int bank = 0; bank < num_banks; ++bank) {
        auto row_list = last_row[bank];
        if (row_list == nullptr) {
            printf("Bank %d: NULL\n", bank);
            continue;
        }
        printf("Bank %d:\n", bank);
        for (auto it : *row_list) {
            if (it != queue->end()) {
                dram_req_t *req = *it;
                if (req && req->data) {
                    printf("  -> ");
                    req->data->print(stdout);
                } else {
                    printf("  -> NULL or missing data\n");
                }
            } else {
                printf("  -> Invalid iterator (end)\n");
            }
        }
    }
}

void fffrfcfs_scheduler::print(FILE *fp) {
    printf("[%d] num_pending: %d: queue_size: %d\n", m_dram->m_gpu->gpu_sim_cycle, num_pending(), m_config->gpgpu_frfcfs_dram_sched_queue_size);
  for (unsigned b = 0; b < m_config->nbk; b++) {
    printf(" %u: queue length = %u\n", b, (unsigned)m_queue[b].size());
    print_queue(&(m_queue[b]), "Read Queue");
    print_bins(&(m_bins[b]), &(m_queue[b]), "Read Bins");
    //TODO:print_last_row(m_last_row, m_config->nbk, &(m_queue[b]), "Last Row Pointer");
  }
}
// } yhyang 250611_15 for scheduling only ndc_fill when rsv_fail_queue is full

