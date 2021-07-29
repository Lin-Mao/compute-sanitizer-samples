/* Copyright (c) 2019-2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <sanitizer_callbacks.h>

// CUDA include for cudaError_t
#include <driver_types.h>

#include <iostream>

static void SANITIZERAPI ApiTrackerCallback(
    void* userdata,
    Sanitizer_CallbackDomain domain, // callback Domain
    Sanitizer_CallbackId cbid, // callback ID
    const void* cbdata) // cbdata is used to pass data to spcific callback function, type may vary.
{
    if (domain == SANITIZER_CB_DOMAIN_DRIVER_API)
        std::cout<< "Calling a CUDA Driver API..." << std::endl;
    if (domain != SANITIZER_CB_DOMAIN_RUNTIME_API)
        return;

    // The type of cbdata is Sanitizer_CallbackData when domains are equal to SANITIZER_CB_DOMAIN_DRIVER_API or SANITIZER_CB_DOMAIN_RUNTIME_API.
    auto* pCallbackData = (Sanitizer_CallbackData*)cbdata;

    // cuda api enter -by lm
    if (pCallbackData->callbackSite == SANITIZER_API_ENTER) {
        std::cout << "API call into " << pCallbackData->functionName << " params address: "
            << pCallbackData->functionParams << std::endl;
    }

    // cuda api exit -by lm
    if (pCallbackData->callbackSite == SANITIZER_API_EXIT)   
    {
        auto returnValue = *(cudaError_t*)pCallbackData->functionReturnValue;
        std::cout << "API call out " << pCallbackData->functionName << " (return code: "
            << returnValue << ")" << std::endl;
    }
}

int InitializeInjection()
{
    Sanitizer_SubscriberHandle handle;

    sanitizerSubscribe(&handle, ApiTrackerCallback, nullptr);
    sanitizerEnableDomain(1, handle, SANITIZER_CB_DOMAIN_RUNTIME_API);

    return 0;
}

int __global_initializer__ = InitializeInjection();
