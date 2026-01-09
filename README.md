# Introduction to Machine Learning


## Requirements for the Milestone #3

- Improvement in comparison to Milestone #2 (3 points)
- Report (2 points)



### Things to cover in code AND report:

1. data augmentation, including quality enhancements: removing silent passages from the training data,
2. choosing the length of a single excerpt, input normalization,
3. changing the number and size of the layers,
4. different optimizers (SGD, Adam), learning rate schedule, weight decay,
5. batch normalization (before vs after activations),
6. skip connections,
7. dropout,
8. Monte Carlo dropout for uncertainty measurement,
9. activation functions (ReLU, sigmoid) tested against diMerent weight,
10. initializations (Xavier, He, uniform)
11. a comparison to pre-trained models (transfer learning, ﬁne-tuning).

Your project needs to cover any eight (8) of the above-listed points to be considered for
5.0 grade,
any ﬁve (5) to be considered for 4.0 grade,
and any three (3) to be considered
for 3.0 grade.


Moreover, **report** should contain a section with description of the work done by each group member.

<br>

### Currently we have: 

- [x] data augmentation, including quality enhancements: removing silent passages from the training data,
- [x] choosing the length of a single excerpt, input normalization,
- [ ] changing the number and size of the layers,
- [x] different optimizers (SGD, Adam), learning rate schedule, weight decay, (partially)
- [ ] batch normalization (before vs after activations),
- [ ] skip connections,
- [x] dropout,
- [x] Monte Carlo dropout for uncertainty measurement,
- [ ] activation functions (ReLU, sigmoid) tested against diMerent weight,
- [ ] initializations (Xavier, He, uniform)
- [ ] a comparison to pre-trained models (transfer learning, ﬁne-tuning).

But keep in mind that many of these require <u> explicit comparison tests</u> and documentation in the report to fully count.



<br><br>

# Workplan

It was generated using AI, therefore please keep in mind that there might be some mistakes.
Also feel free to modify it. For now it's just a rough estimation what we can do.



## 1. Requirement Completion

Current status: ~5 points implemented in code. We need 3 more + documentation/comparison for all. (Target: 8 points for 5.0)

- [ ] Point 3: Layer Scaling Comparison
    - [ ] Modify `Backbone` to easily swap the number of CNN filters or GRU hidden units.
    - [ ] Run at least two variations (e.g., "Lightweight" vs "Deep") and compare performance.
- [ ] Point 4: Optimizer & Schedule Comparison
    - [ ] You have SGD and AdamW implemented. You must run a test comparing them.
    - [ ] Document the effect of the `CosineAnnealingLR` schedule vs. a fixed learning rate.
- [ ] Point 5: Batch Normalization Placement
    - [ ] Current code places BN **before** ReLU.
    - [ ] Create a version with BN **after** the activation function.
    - [ ] Run a comparison test and document which performed better.
- [ ] Point 9: Activation Function Testing
    - [ ] Compare the current ReLU/Sigmoid mix against a version using LeakyReLU or ELU.
    - [ ] Document the impact on training stability.
- [ ] Point 10: Initialization Comparison
    - [ ] Current `AAMSoftmax` uses `xavier_uniform_`. 
    - [ ] Test the `Backbone` layers with `He` (Kaiming) vs. `Uniform` initializations.
- [ ] Point 11: Transfer Learning (High Impact)
    - [ ] Implement a comparison against a pre-trained model (e.g., a pre-trained ResNet or VGG adapted for audio/spectrograms).




## 2. Report

- [ ] LaTeX: I suggest we use it for the report.
- [ ] Work Distribution Section: Create a table or list describing exactly what each group member contributed.
- [ ] Comparison Analysis: For every point selected (e.g., Optimizers, Dropout), include:
    - [ ] Accuracy/Loss plots comparing the two methods.
    - [ ] A brief analysis of why one performed better.
- [ ] Quality Enhancements: Document the "Silence Removal" process and how it improved the training data quality.
- [ ] Normalization & Chunking: Explain the choice of 1.0s excerpts and volume normalization parameters.
- [ ] TensorBoard Exports: Export the charts from your `SummaryWriter` logs to include in the report.
- [ ] Cleanup: Ensure the `in_depth_eda.ipynb` findings (like SNR and silence ratios) are mentioned in the report to justify your preprocessing choices.

<br>




## Split the work:

### Aleksander
### Michał
### Piotr
### Rafał
### Mantas
