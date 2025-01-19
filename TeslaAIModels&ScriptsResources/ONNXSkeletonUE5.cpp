//
//  ONNXSkeletonUE5.cpp
//  
//
//  Created by Gunnar Beck on 1/10/25.
//

#include <stdio.h>
#include "YourProject.h" //create custom header file
#include "ONNXRuntime.h" //create custom header file
#include "Engine/TextureRenderTarget2D.h" //a headerfile in UE5 that may indicate the target
#include "Runtime/Engine/Classes/Engine/Texture2D.h" //a headerfile in UE5 that may indicate the texture
#include "Camera/CameraComponent.h" //a headerfile in UE5 that may indicate the camera
#include <onnxruntime/core/session/onnxruntime_cxx_api.h> //api call

using namespace std; // Use the standard namespace for convenience

//starting our class
class FONNXModel
{
public:
    //constructor
    FONNXModel(const FString& ModelPath);
    TArray<float> RunInference(const TArray<float>& InputData);

private:
    Ort::Env Env;
    Ort::SessionOptions SessionOptions;
    unique_ptr<Ort::Session> Session;
};

// Constructor to initialize ONNX model
FONNXModel::FONNXModel(const FString& ModelPath)
    : Env(ORT_LOGGING_LEVEL_WARNING, "ONNXRuntime")
{
    SessionOptions.SetIntraOpNumThreads(1);
    SessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);
    Session = make_unique<Ort::Session>(Env, TCHAR_TO_UTF8(*ModelPath), SessionOptions);
}

// Run inference with the model
TArray<float> FONNXModel::RunInference(const TArray<float>& InputData)
{
    // Prepare input tensor
    Ort::AllocatorWithDefaultOptions Allocator;
    vector<int64_t> InputShape = { 1, 3, 224, 224 }; // Example for a 3x224x224 image
    size_t InputTensorSize = 3 * 224 * 224;

    // Create input tensor
    vector<float> InputTensorValues(InputData.GetData(), InputData.GetData() + InputTensorSize);
    auto MemoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value InputTensor = Ort::Value::CreateTensor<float>(MemoryInfo, InputTensorValues.data(), InputTensorSize, InputShape.data(), InputShape.size());

    // Get model outputs
    auto OutputNames = Session->GetOutputNames(Allocator);
    auto OutputInfo = Session->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo();
    vector<int64_t> OutputShape = OutputInfo.GetShape();
    size_t OutputTensorSize = 1;

    for (auto Dim : OutputShape)
        OutputTensorSize *= Dim;

    vector<float> OutputTensorValues(OutputTensorSize);
    Ort::Value OutputTensor = Ort::Value::CreateTensor<float>(MemoryInfo, OutputTensorValues.data(), OutputTensorSize, OutputShape.data(), OutputShape.size());

    // Run inference
    Session->Run(Ort::RunOptions{ nullptr }, &OutputNames[0], &InputTensor, 1, &OutputNames[0], &OutputTensor, 1);

    return TArray<float>(OutputTensorValues.begin(), OutputTensorValues.end());
}

// Using the ONNX model in a UE5 Camera
void ProcessCameraFeed(UCameraComponent* Camera, FONNXModel& Model)
{
    // Get camera feed as a texture
    UTextureRenderTarget2D* RenderTarget = Camera->GetRenderTarget();
    FTextureRenderTargetResource* RenderTargetResource = RenderTarget->GameThread_GetRenderTargetResource();
    TArray<FColor> PixelData;

    // Read pixels from render target
    RenderTargetResource->ReadPixels(PixelData);

    // Preprocess the image into ONNX input format
    TArray<float> InputData;
    for (const FColor& Pixel : PixelData)
    {
        InputData.Add(Pixel.R / 255.0f);
        InputData.Add(Pixel.G / 255.0f);
        InputData.Add(Pixel.B / 255.0f);
    }

    // Resize input to match model requirements (e.g., 3x224x224)
    // Use your preferred image preprocessing method here

    // Run inference with the ONNX model
    TArray<float> InferenceResult = Model.RunInference(InputData);

    // Process results (e.g., bounding boxes, segmentation)
    // Render results to the viewport or use for game logic
}
