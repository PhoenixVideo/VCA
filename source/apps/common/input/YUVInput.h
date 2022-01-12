/*****************************************************************************
 * Copyright (C) 2021 Christian Doppler Laboratory ATHENA
 *
 * Authors: Steve Borho <steve@borho.org>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.
 *****************************************************************************/

#pragma once

#include "IInputFile.h"

#include <fstream>

namespace vca {

class YUVInput : public IInputFile
{
public:
    YUVInput() = delete;
    YUVInput(std::string &fileName, vca_frame_info &openFrameInfo, unsigned skipFrames);
    ~YUVInput() = default;

    bool readFrame(frameWithData &frame) override;
};

} // namespace vca