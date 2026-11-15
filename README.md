# Pet Breed Classifier (Oxford-IIIT Pets)

Fine-tuned a pretrained ResNet-18 to classify 37 cat and dog breeds, reaching **87.6% accuracy on a held-out test set**. The project compares a small CNN trained from scratch with transfer learning, and looks at which breeds the final model confuses.

## Data

- Oxford-IIIT Pet Dataset: 37 breeds, 3,680 images in the `trainval` split and 3,669 in the `test` split
- `trainval` was split 80/20 into 2,944 training and 736 validation images, using a fixed random seed so the validation set is the same in every script
- The test set was left untouched until the final evaluation
- Images were resized to 224x224 and normalised with the ImageNet mean and standard deviation, which is what the pretrained ResNet expects

## Experiments

All scores below are accuracy on the 736-image validation set unless stated otherwise.

| Model | Trainable parameters | Validation accuracy |
|---|---|---|
| Random guessing | - | about 2.7% |
| Small CNN from scratch (10 epochs) | millions (first dense layer alone has about 6.4M) | 11.1% (best 13.0% at epoch 3); training accuracy reached 100% |
| ResNet-18, last layer only (5 epochs) | 18,981 | about 86-88% |
| ResNet-18, last block (`layer4`) and last layer unfrozen (5 epochs) | 8,412,709 | 89.9% for the saved model |

The scratch CNN's validation loss climbed from 3.25 (epoch 2) to over 10 while its training loss fell to 0.000, a clear sign of overfitting. Another run of the unfrozen-`layer4` recipe reached 91.4% validation accuracy, so the validation set is noisy at the 1-2 point level; the table reports the model that was actually saved and tested.

## Final result

On the 3,669-image test set, used once after the model was chosen:

- **Test accuracy: 87.6%** (test loss 0.407)
- With a test set this size the score has a margin of roughly 1 point either way.
- It is lower than the 89.9% validation score. The model was chosen using the validation set, which makes that score slightly optimistic, and the test set was not involved in any decision.

## Mistakes

I looked at the 74 validation photos (out of 736) that the final model got wrong and counted which breeds were confused with which. The most common pairs were:

| Count | True breed | Predicted as |
|---|---|---|
| 4 | Beagle | Basset Hound |
| 3 | Egyptian Mau | Bengal |
| 2 | English Cocker Spaniel | English Setter |
| 2 | American Pit Bull Terrier | Staffordshire Bull Terrier |
| 2 | Birman | Siamese |
| 2 | Ragdoll | Birman |

The ten most common pairs also include American Bulldog to Beagle, Ragdoll to Persian, British Shorthair to Russian Blue and Russian Blue to Bombay (2 each).

Most of the top pairs are breeds that look similar: hounds and spaniels, spotted cats, long-haired cats with pale coats. Every one of the top ten pairs stays within a species (cat for cat, dog for dog), so the model separates cats from dogs well and struggles mainly with fine differences between near-identical breeds. One pair surprised me: American Bulldog predicted as Beagle, which look quite different, so it may come from unusual photos or crops.

These are weak patterns. The ten most common pairs account for 23 of the 74 mistakes, each pair happened only 2 to 4 times, and the other 51 mistakes were scattered one-offs. A single odd photo could create a pair, so this shows what kinds of errors occur and is not a ranking of the model's weaknesses. The analysis used the validation set; the test set was only used once, for the final accuracy.

## What I learned

A small CNN trained from scratch memorized its 2,944 training photos (100% training accuracy) but reached only 11% on validation photos, and its validation loss climbed above 10. A pretrained ResNet-18, which had already learned edges, textures and shapes from a much larger dataset, reached about 86% after training only its final layer, so transfer learning works well when the dataset for the new task is small.

## How to run

1. Install the dependencies: `pip install torch torchvision`. For an NVIDIA GPU, use the CUDA build of PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/); the default pip install on Windows is CPU-only.
2. Download the dataset from the [Oxford-IIIT Pet Dataset page](https://www.robots.ox.ac.uk/~vgg/data/pets/) and extract `images` and `annotations` so the layout is `data/oxford-iiit-pet/images/` and `data/oxford-iiit-pet/annotations/`. The loaders use `download=False`.
3. Run the scripts in this order:
   - `scratch_cnn.py`: small CNN trained from scratch
   - `finetune_head.py`: pretrained ResNet-18, last layer only
   - `finetune_layer4.py`: unfreezes the last block, trains, and saves `pets_resnet18.pt`
   - `load_model.py`: reloads the saved model, prints validation accuracy, the most common mistakes, and the final test accuracy (run the test step once)

`loaders.py` (data pipeline and splits) and `train_utils.py` (the shared training and evaluation loop) are imported by the scripts above. The dataset and the saved model file are not included in this repository.

## Credits

- Dataset: Oxford-IIIT Pet Dataset by O. M. Parkhi, A. Vedaldi, A. Zisserman and C. V. Jawahar (Visual Geometry Group, University of Oxford), released under CC BY-SA 4.0. The copyright of the images remains with their original owners.
- Model: ResNet-18 with ImageNet pretrained weights from torchvision.
