
#include "Analyzer.h"

#include <string>

namespace vca {

Analyzer::Analyzer(vca_param cfg)
{
    this->cfg      = cfg;
    auto nrThreads = cfg.nrFrameThreads * cfg.nrSliceThreads;
    log(cfg, LogLevel::Info, "Starting " + std::to_string(nrThreads) + " threads");
    for (unsigned i = 0; i < nrThreads; i++)
        this->threadPool.emplace_back(this->cfg, this->jobs, this->results, i);
}

Analyzer::~Analyzer()
{
    for (auto &thread : this->threadPool)
        thread.abort();
    this->jobs.abort();
    this->results.abort();
    for (auto &thread : this->threadPool)
        thread.join();
}

vca_result Analyzer::pushFrame(vca_frame *frame)
{
    if (!this->checkFrameSize(frame->info))
        return vca_result::VCA_ERROR;

    Job job;
    job.frame = frame;
    job.jobID = this->frameCounter;
    // job.macroblockRange = TODO

    this->jobs.push(job);
    this->frameCounter++;

    return vca_result::VCA_OK;
}

bool Analyzer::resultAvailable()
{
    return this->results.empty();
}

vca_result Analyzer::pullResult(vca_frame_results *outputResult)
{
    auto result = this->results.waitAndPop();

    if (!result)
        return vca_result::VCA_ERROR;

    outputResult->poc           = result->poc;
    outputResult->averageEnergy = result->averageEnergy;
    std::memcpy(outputResult->energyPerBlock,
                result->energyPerBlock.data(),
                result->energyPerBlock.size() * sizeof(int32_t));

    return vca_result::VCA_OK;
}

bool Analyzer::checkFrameSize(vca_frame_info frameInfo)
{
    if (!this->frameInfo)
    {
        if (frameInfo.bitDepth < 8 || frameInfo.bitDepth > 16)
        {
            log(this->cfg,
                LogLevel::Error,
                "Frame with invalid bit " + std::to_string(frameInfo.bitDepth) + " depth provided");
            return false;
        }
        if (frameInfo.width == 0 || frameInfo.width % 2 != 0 || frameInfo.height == 0
            || frameInfo.height % 2 != 0)
        {
            log(this->cfg,
                LogLevel::Error,
                "Frame with invalid size " + std::to_string(frameInfo.width) + "x"
                    + std::to_string(frameInfo.height) + " depth provided");
            return false;
        }
        this->frameInfo = frameInfo;
    }

    if (frameInfo.bitDepth != this->frameInfo->bitDepth || frameInfo.width != this->frameInfo->width
        || frameInfo.height != this->frameInfo->height
        || frameInfo.colorspace != this->frameInfo->colorspace)
    {
        log(this->cfg, LogLevel::Error, "Frame with different settings revieved");
        return false;
    }

    return true;
}

} // namespace vca