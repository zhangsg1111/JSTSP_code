1. **Download the pretrained model weights** from the following links:

   - **ViT-Base (patch size 16, input resolution 224)**
      https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth
   - **ViT-Large (patch size 16, input resolution 224)**
      https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_224-4ee7a4dc.pth

2. **Convert the pretrained weights** by running the notebook:

   ```
   vit_trans_pretrained_weight.ipynb
   ```

3. **Move the converted pretrained weights** to the following directory:

   ```
   ./pretrained_weights/models--vision-transformer/
   ```

4. **Runing the Main file**

```
EfficientSplitedLLM-vit_fwc-bwc_v3-tinyimagenet.ipynb
or
EfficientSplitedLLM-vit_large_fwc-bwc_v3-cifar100-Copy1.ipynb
```