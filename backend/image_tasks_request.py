{
    "taskType": "TEXT_IMAGE",
    "textToImageParams": {
        "text": string,
        "negativeText": string
    },
    "imageGenerationConfig": {
        "width": int,
        "height": int,
        "quality": "standard" | "premium",
        "cfgScale": float,
        "seed": int,
        "numberOfImages": int
    }
}
          

{
    "taskType": "TEXT_IMAGE",
    "textToImageParams": {
        "conditionImage": string (Base64 encoded image),
        "controlMode": "CANNY_EDGE" | "SEGMENTATION", 
        "controlStrength": float,
        "text": string,
        "negativeText": string
    },
    "imageGenerationConfig": {
        "width": int,
        "height": int,
        "quality": "standard" | "premium",
        "cfgScale": float,
        "seed": int,
        "numberOfImages": int
    }
}

          
{
    "taskType": "COLOR_GUIDED_GENERATION",
    "colorGuidedGenerationParams": {
        "colors": string[] (list of hexadecimal color values),
        "referenceImage": string (Base64 encoded image),
        "text": string,
        "negativeText": string
    },
    "imageGenerationConfig": {
        "width": int,
        "height": int,
        "quality": "standard" | "premium",
        "cfgScale": float,
        "seed": int,
        "numberOfImages": int
    }
}

{
    "taskType": "IMAGE_VARIATION",
    "imageVariationParams": {
        "images": string[] (list of Base64 encoded images),
        "similarityStrength": float,
        "text": string,
        "negativeText": string 
    },
    "imageGenerationConfig": {
        "numberOfImages": int,
        "height": int,
        "width": int,
        "cfgScale": float,
        "seed": int,
        "numberOfImages": int
    }
}

{
    "taskType": "INPAINTING",
    "inPaintingParams": {
        "image": string (Base64 encoded image),
        "maskPrompt": string,
        "maskImage": string (Base64 encoded image),
        "text": string,
        "negativeText": string
    },
    "imageGenerationConfig": {
        "numberOfImages": int,
        "quality": "standard" | "premium",
        "cfgScale": float,
        "seed": int
    }
}

{
    "taskType": "OUTPAINTING",
    "outPaintingParams": {
        "image": string (Base64 encoded image),
        "maskPrompt": string,
        "maskImage": string (Base64 encoded image),
        "outPaintingMode": "DEFAULT" | "PRECISE",
        "text": string,
        "negativeText": string
    },
    "imageGenerationConfig": {
        "numberOfImages": int,
        "quality": "standard" | "premium"
        "cfgScale": float,
        "seed": int
    }
}

{
    "taskType": "BACKGROUND_REMOVAL",
    "backgroundRemovalParams": {
        "image": string (Base64 encoded image)
    }
}

