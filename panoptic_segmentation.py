class PanopticSegmentationModel:
  def __init__(self, image_path, image_name, image_path_local):    
    self.im_name = image_name
    if not image_path_local:
      self.im = Image.open(requests.get(image_path, stream=True).raw)
    else:
      self.im = im = Image.open(image_path)

    
    # standard PyTorch mean-std input image normalization
    self.transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

  def UnNormalize(self):  
    class UnNormalize(object):
      def __init__(self, mean, std):
        self.mean = mean
        self.std = std

      def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor
    return UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))


  def segmentation_predictions(self):
    self.model, self.postprocessor = torch.hub.load('facebookresearch/detr', 'detr_resnet101_panoptic', pretrained=True, return_postprocessor=True, num_classes=250)
    self.model.eval();
    # mean-std normalize the input image (batch-size: 1)
    self.img = self.transform(self.im).unsqueeze(0)
    self.img1 = self.transform(self.im)
    #img1 = np.array(img1, dtype=np.uint8).copy()
    self.out = self.model(self.img)
  
  def segmentation_plots(self, confidence_threshold):
    scores = self.out["pred_logits"].softmax(-1)[..., :-1].max(-1)[0]
    # threshold the confidence
    keep = scores > confidence_threshold
    num_predictions = keep.sum().item()
    # Plot all the remaining masks
    if (num_predictions > 4):
      ncols = 5
      nrows=math.ceil(num_predictions / ncols)
      fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(18, 10))
      for line in axs:
          for a in line:
              a.axis('off')
      for i, mask in enumerate(self.out["pred_masks"][keep]):
        ax = axs[i // ncols, i % ncols]
        ax.title.set_text('segment id: ' + str(i))
        ax.imshow(mask, cmap="cividis")
        ax.axis('off')
        fig.tight_layout()
    elif (num_predictions > 1):
      ncols= num_predictions
      nrows = 1
      fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(18, 10))
      for i, mask in enumerate(self.out["pred_masks"][keep]):
        ax = axs[i]
        ax.title.set_text('segment id: ' + str(i))
        ax.imshow(mask, cmap="cividis")
        ax.axis('off')
        fig.tight_layout()
    else:
      ncols= num_predictions
      nrows = 1
      fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(18, 10))
      for i, mask in enumerate(self.out["pred_masks"][keep]):
        ax = axs
        ax.title.set_text('segment id: ' + str(i))
        ax.imshow(mask, cmap="cividis")
        ax.axis('off')
        fig.tight_layout()

  def build_segmentation_array(self):      
    result = self.postprocessor(self.out, torch.as_tensor(self.img.shape[-2:]).unsqueeze(0))[0]
    # The segmentation is stored in a special-format png
    panoptic_seg = Image.open(io.BytesIO(result['png_string']))
    self.panoptic_seg = np.array(panoptic_seg, dtype=np.uint8).copy()
    # We retrieve the ids corresponding to each mask
    self.panoptic_seg_id = rgb2id(self.panoptic_seg)
    self.panoptic_seg[:, :, :] = 0

  def set_panoptic_segmentation_range(self, segmentation_range):
    #self.start_id = start_id
    #self.end_id = end_id
    self.segmentation_range = segmentation_range
  
  def obtain_segmented_masks(self):  
    unorm = self.UnNormalize()
    self.img1 = unorm(self.img1)
    self.img1 = T.ToPILImage()(self.img1).convert("RGB")
    self.img1 = np.array(self.img1, dtype=np.uint8).copy()
    for id in self.segmentation_range:
      self.panoptic_seg[self.panoptic_seg_id == id] = 255
      new_seg_image_gray = cv2.cvtColor(self.panoptic_seg, cv2.COLOR_RGB2GRAY) # Convert the mask to grayscale
      plt.figure(figsize=(15,15))
      plt.imshow(self.panoptic_seg)
      plt.axis('off')
      plt.show()
      masked_out = np.array(self.img1, dtype=np.uint8).copy()
      masked_out[np.where(self.panoptic_seg != 255)] = 255
      self.masked_out = masked_out
      #panoptic_seg_gray = cv2.cvtColor(self.panoptic_seg, cv2.COLOR_RGB2GRAY) # Convert the mask to grayscale
      #masked_out = cv2.bitwise_and(self.img1, self.img1, mask=panoptic_seg_gray) # Blend the mask
      #self.masked_out_new = np.where(masked_out != 0, masked_out, 255) # Remove the background      
      plt.figure(figsize=(15,15))
      plt.imshow(self.masked_out)
      plt.axis('off')
      plt.show()

  def save_segmented_image(self):  
    final_segmented_image = Image.fromarray(self.masked_out)
    final_segmented_image.save(os.path.join(sys_path + "Data/Segmented Content Images/" + self.im_name + ".png"))
    #stylized_image = Image.open(results_path + "0.png")
  
  def change_base_image(self, base_image):
    self.img1 = base_image
    self.img1 = self.transform(self.img1)
    unorm = self.UnNormalize()
    self.img1 = unorm(self.img1)
    self.img1 = T.ToPILImage()(self.img1).convert("RGB")

  def blend_segmented_mask(self, stylized_image):
    original_image_array = np.array(self.img1, dtype=np.uint8).copy()
    original_image_array[np.where(self.panoptic_seg == 255)] = self.panoptic_seg[np.where(self.panoptic_seg == 255)]
    original_image_segment_removed = Image.fromarray(original_image_array)
    length, width = stylized_image.size
    original_image_segment_removed = original_image_segment_removed.resize(stylized_image.size, Image.ANTIALIAS)
    original_image_segment_removed_array = np.array(original_image_segment_removed)
    stylized_image_array = np.array(stylized_image)
    original_image_segment_removed_array[np.where(original_image_segment_removed_array == 255)] = stylized_image_array[np.where(original_image_segment_removed_array == 255)]
    final_image = Image.fromarray(original_image_segment_removed_array)
    display.display(final_image)          
    return final_image          
  
  def return_current_mask(self):
    return self.panoptic_seg

  def run(self):  
    self.segmentation_predictions()
    self.segmentation_plots(confidence_threshold = 0.85)
    self.build_segmentation_array()
    print('In order to obtain a segmented mask, please run the "self.obtain_segmented_masks()" function and input the starting and ending segmentation ids that were given in the plots.')
    self.obtain_segmented_masks()
    self.save_segmented_image()
    return self.masked_out
