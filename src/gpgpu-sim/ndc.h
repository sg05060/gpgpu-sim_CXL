// Copyright (c) 2009-2011, Tor M. Aamodt, Ivan Sham, Ali Bakhoda, 
// George L. Yuan, Wilson W.L. Fung
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

#ifndef NDC_H
#define NDC_H

#include "dram.h"
#include "gpu-cache.h"
#include <queue>

// pshyun {
// moved from l2cache.h. It is required for making new mem_fetch by NDC when miss alloc
class ndc_mf_allocator : public mem_fetch_allocator {
   public:
    ndc_mf_allocator(const memory_config *config) { 
        m_memory_config = config; 
    }
    virtual mem_fetch *alloc(const class warp_inst_t &inst,
                           const mem_access_t &access,
                           unsigned long long cycle) const {
        abort();
        return NULL;
    }
    virtual mem_fetch *alloc(new_addr_type addr, mem_access_type type,
                            unsigned size, bool wr, unsigned long long cycle,
                            unsigned long long streamID) const;

    virtual mem_fetch *alloc(new_addr_type addr, mem_access_type type,
                            const active_mask_t &active_mask,
                            const mem_access_byte_mask_t &byte_mask,
                            const mem_access_sector_mask_t &sector_mask,
                            unsigned size, bool wr, unsigned long long cycle,
                            unsigned wid, unsigned sid, unsigned tpc,
                            mem_fetch *original_mf,
                            unsigned long long streamID) const;
    private:
        const memory_config *m_memory_config;
};
// } pshyun

class ndc_t : public dram_t {
   public:
    // psh: dram_t에 따라 gpgpu_sim *gpu추가 -> 나중에 상위단에서 instance할때 Check
    //ndc_t(unsigned int partition_id, const struct memory_config *config, class memory_stats_t *stats, class memory_partition_unit *mp);
    ndc_t(unsigned int partition_id, const struct memory_config *config, 
            class memory_stats_t *stats, class memory_partition_unit *mp,
            class gpgpu_sim *gpu);
    virtual ~ndc_t();
    void cycle() override;  // DRAM의 사이클 동작 오버라이딩
    // void push_to_cache(mem_fetch *mf);

    // pshyun {
    //cache_request_status access_cache(new_addr_type addr, mem_fetch *mf, unsigned time, std::list<cache_event> &events);
    bool full_from_cxl() const;
    void push_from_cxl(class mem_fetch *data);
    void cache_cycle();
    // } pshyun

    // pshyun:private:
   public:
    l3_ndc_cache *m_ndc_cache;  // 기존 data_cache 구조를 재활용
    // pshyun {
    void scheduler_fffrfcfs();
    ndc_mf_allocator *m_mf_allocator;
    void ndc_access(dram_req_t *cmd, bool from_rsvq);
    bool ndc_fill_access(dram_req_t *cmd);

    void accumulate_L3_NDCcache_stats(class cache_stats &l3_ndc_stats) const;
    void get_L3_NDCcache_sub_stats(struct cache_sub_stats &css) const;

    fifo_pipeline<dram_req_t> *fill_mrqq;
    fifo_pipeline<dram_req_t> *dram_ndc_fill_queue;
    fifo_pipeline<dram_req_t> *ndc_rsv_fail_queue;

    class L3_NDC_interface *m_L3_NDC_interface;
    friend class m_L3_NDC_interface;
    int err_cnt;
    //  } pshyun
};

class L3_NDC_interface : public mem_fetch_interface {
   public:
    L3_NDC_interface(/*memory_sub_partition*/ ndc_t *unit) { m_unit = unit; }
    virtual ~L3_NDC_interface() {}
    virtual bool full(unsigned size, bool write) const {
        // assume read and write packets all same size
        // return m_unit->m_L2_dram_queue->full();
        return m_unit->returnq_full();
    }
    virtual void push(mem_fetch *mf) {
        mf->set_status(IN_PARTITION_MC_RETURNQ, 0 /* TODO:gpu_sim_cycle+gpu_tot_sim_cycle*/);
        printf("[PSH_DEBUG][ndc->returnq push] uid : %d, request_size : %d\n", mf->get_request_uid(), mf->get_access_size());
        m_unit->returnq->push(mf);
    }

   private:
    // memory_sub_partition *m_unit;
    ndc_t *m_unit;
};

#endif /*NDC_H*/
