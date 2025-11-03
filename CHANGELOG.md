# Changelog

## [0.9.0](https://github.com/chanzuckerberg/saber/compare/saber-em-v0.8.0...saber-em-v0.9.0) (2025-11-01)


### ✨ Features

* Add a web gui ([b3bdd7a](https://github.com/chanzuckerberg/saber/commit/b3bdd7a9c02b3eb413074a5c83893d9d1c9e84e3))
* instance multi-slab 3D seg, storage of AMG parameters ([9fa4ff6](https://github.com/chanzuckerberg/saber/commit/9fa4ff6bb067d2f69e7ea762c696cb3e80438f7f))
* major restructure to 3D segmentation workflow ([66957ea](https://github.com/chanzuckerberg/saber/commit/66957ea15e66916ca87d919681518e95bc047450))


### 🐞 Bug Fixes

* make sure amg cfg is properly allocated during inference ([9af571c](https://github.com/chanzuckerberg/saber/commit/9af571c67282f95e5d79fd62b4383902d1f76679))
* make sure amg cfg is properly allocated during inference ([25d3eac](https://github.com/chanzuckerberg/saber/commit/25d3eaccd6690e0796d8fa3f57cca72a1bed3165))
* Make sure AMG Parameters are properly read for Tomography Workflow ([d167af2](https://github.com/chanzuckerberg/saber/commit/d167af243e0f1e7c72dc439dbfc435c5248429db))
* make sure amg parameters are properly referenced for tomo workflow ([bc8a1b6](https://github.com/chanzuckerberg/saber/commit/bc8a1b6a06fd3746474883b4ecd604688bad62cc))

## [0.8.0](https://github.com/chanzuckerberg/saber/compare/saber-em-v0.7.0...saber-em-v0.8.0) (2025-10-03)


### ✨ Features

* 3D annotation gui and tracking of metadata for zarr files ([2253097](https://github.com/chanzuckerberg/saber/commit/2253097524da325a6ee2f8ff189de39e63d258a5))
* Interactivity improvements for the annotation gui ([966df35](https://github.com/chanzuckerberg/saber/commit/966df354b045d0bac79d1be29954bd6c2fc43a6b))
* run 2D preprocessing on tif stack ([3c75d77](https://github.com/chanzuckerberg/saber/commit/3c75d7700a7259c851df22cbcf8a7d7597d6d68c))
* vcp dataio ([6d227b4](https://github.com/chanzuckerberg/saber/commit/6d227b4e59c3212f010be644f94c84876fe6cb2a))
* workflow for fib segmentation ([d53b297](https://github.com/chanzuckerberg/saber/commit/d53b29726981e2bd0f6c212b531276e8d23dadee))


### 🐞 Bug Fixes

* feature improvements to the 3D gui. Ability to rotate volume. Faster reading. Linking of images ([bda2cc1](https://github.com/chanzuckerberg/saber/commit/bda2cc13a25ac492f27e4b2a7f4f598f8ffc5b7a))
* Gui3d ([0fff24d](https://github.com/chanzuckerberg/saber/commit/0fff24db4aba9ce3c25fbb80768a996797ad871f))
* improve the speed for fib segmentation, ensure correct format is saved for vcp web gui ([680342a](https://github.com/chanzuckerberg/saber/commit/680342a9dc0ad93a98c348e548a8a2516eaa8d5f))
* improved workflow for fib segmentation ([e028e62](https://github.com/chanzuckerberg/saber/commit/e028e62ae91e5897ab463eb14a7cc35c0e16554f))
* speed up the loading time between frames ([3758068](https://github.com/chanzuckerberg/saber/commit/37580687df73495fd052d1a16c803ad9acdc52a8))

## [0.7.0](https://github.com/chanzuckerberg/saber/compare/saber-em-v0.6.1...saber-em-v0.7.0) (2025-08-19)


### ✨ Features

* text annotation-gui ([c4f9ded](https://github.com/chanzuckerberg/saber/commit/c4f9ded3414e2896da588303d5d30c21e73df4df))

## [0.6.1](https://github.com/chanzuckerberg/saber/compare/saber-em-v0.6.0...saber-em-v0.6.1) (2025-08-07)


### 🐞 Bug Fixes

* versioning ([55882a4](https://github.com/chanzuckerberg/saber/commit/55882a4d938dce8075f25e764dc2197d25eb075a))
* versioning ([fc672ca](https://github.com/chanzuckerberg/saber/commit/fc672cabfd756b5775e6b41d138eed2857ed254a))

## [0.6.0](https://github.com/chanzuckerberg/saber/compare/saber-em-v0.5.0...saber-em-v0.6.0) (2025-08-07)


### ✨ Features

* add release-please configuration ([16da577](https://github.com/chanzuckerberg/saber/commit/16da5771c2ba345ba0b2ebd0ec6d7e5e63280da4))
* add release-please configuration ([6e8d558](https://github.com/chanzuckerberg/saber/commit/6e8d55899533ed950433fee339c951a9a31d59b9))
* Image Saving and SAM2 Mask Batching ([df4ab57](https://github.com/chanzuckerberg/saber/commit/df4ab57d693af2451416ab5306ed596e936e5827))
* prepare for v0.5.0 release ([6cafc31](https://github.com/chanzuckerberg/saber/commit/6cafc31c4f3a5f71b490779aaed795a0f394ef8e))


### 🐞 Bug Fixes

* add README to PyPI description ([d9556fb](https://github.com/chanzuckerberg/saber/commit/d9556fb25453f2f15fbada3e1ccb51696d4aed4f))
* dynamic versioning ([cf280dd](https://github.com/chanzuckerberg/saber/commit/cf280dd2dca03eb645e5bd7fa6d1b5d554173cc4))
* dynamic versioning ([d1a67c8](https://github.com/chanzuckerberg/saber/commit/d1a67c808a05e606718624d587279260d12fe289))
* make sure inference state only read once ([f6963e7](https://github.com/chanzuckerberg/saber/commit/f6963e7545735ddbcfd46bd55048881e5072cc7a))
* make sure inference state only read once ([3672dce](https://github.com/chanzuckerberg/saber/commit/3672dce272d990010ce6f3c5d777e02e703d4ef7))
* make sure packages get submitted to pypi ([231361e](https://github.com/chanzuckerberg/saber/commit/231361eebab8a35b42e6f1466c5439a357f8b6df))
* make sure search pattern is correct ([2ffbb53](https://github.com/chanzuckerberg/saber/commit/2ffbb53c9aa64528095e331cd3203f5d151a1a78))
* release-please ([10e89a6](https://github.com/chanzuckerberg/saber/commit/10e89a6a06850e29a3d196a421ae3c4d84f25324))
* release-please ([90052a3](https://github.com/chanzuckerberg/saber/commit/90052a3984eddad8314f35f5ec158fb54dbd621f))
* rename release-please config file ([9403ffb](https://github.com/chanzuckerberg/saber/commit/9403ffbb3d293bc2b171828df81c8196ad977cb8))
* rename release-please config file ([03fe8c2](https://github.com/chanzuckerberg/saber/commit/03fe8c2f6df1b80b48d4ac696dbde85dc507eed0))
* resolve release please CI ([43a6847](https://github.com/chanzuckerberg/saber/commit/43a68470b266cfb9583be17d926549af7af7b09e))
* resolve release please CI ([dd26ff5](https://github.com/chanzuckerberg/saber/commit/dd26ff5230c636a61d5c34dff28d5fcfd554e1d0))
* use PAT for release-please to bypass org permissions ([c135a97](https://github.com/chanzuckerberg/saber/commit/c135a97bd76cfdcbf2a2edfa5ea38ed97bc3be3f))
* use PAT for release-please to bypass org permissions ([24b18d3](https://github.com/chanzuckerberg/saber/commit/24b18d30027315c9007d9db38b6b8637a3832a55))
* use the organization token ([b4b9d11](https://github.com/chanzuckerberg/saber/commit/b4b9d1179817c79b48af7b623fc1a9c82a9ca88d))
* versioning ([9710436](https://github.com/chanzuckerberg/saber/commit/9710436a68dd9f7484dedba78a9c70a60b0fa9d4))
* versioning ([4f54535](https://github.com/chanzuckerberg/saber/commit/4f545351cb0e169b369d002e21725c8900fc1c6d))


### 📝 Documentation

* better description for importing data to copick ([e5e9b64](https://github.com/chanzuckerberg/saber/commit/e5e9b64c514e9b8f5d7172d39dcb8f0e6b6d35c6))


### 🧹 Miscellaneous Chores

* CCIE-4984 conform to open sourcing guidelines ([0badd9c](https://github.com/chanzuckerberg/saber/commit/0badd9c579bd2cf94f57d977c482343bb5e79206))
* CCIE-4984 conform to open sourcing guidelines ([34fdf98](https://github.com/chanzuckerberg/saber/commit/34fdf9885140778f57667672358458744b6581cd))
* **main:** release saber 0.3.0 ([0121f88](https://github.com/chanzuckerberg/saber/commit/0121f887643c6e78d7f619d86981c6bac4adda62))
* **main:** release saber 0.3.0 ([d94c1ed](https://github.com/chanzuckerberg/saber/commit/d94c1eda84bf0f14a001a2c2ee9d1017d8ee49b5))
* **main:** release saber 0.6.0 ([a47a28a](https://github.com/chanzuckerberg/saber/commit/a47a28a616f09656e953483fac78a87dadda7da7))
* **main:** release saber 0.6.0 ([c043558](https://github.com/chanzuckerberg/saber/commit/c04355877191d04696633504f24da456cad81e73))
* **main:** release saber 0.7.0 ([99b3ddb](https://github.com/chanzuckerberg/saber/commit/99b3ddb1e3f8ecbeee5ab4b513761d847edbadfa))
* **main:** release saber 0.7.0 ([824ca17](https://github.com/chanzuckerberg/saber/commit/824ca172ea2b3e685b6082e5ce61dd173cb86ca8))
* **main:** release saber 0.7.1 ([210ee32](https://github.com/chanzuckerberg/saber/commit/210ee32d794da56dfd840849be9899f2f4079417))
* **main:** release saber 0.7.1 ([7c79b94](https://github.com/chanzuckerberg/saber/commit/7c79b9452e340c402a54019910437d38a8f4b807))
* **main:** release saber 0.7.2 ([3bee497](https://github.com/chanzuckerberg/saber/commit/3bee4971361a25f39f64df662ff9634fac5201a2))
* **main:** release saber 0.7.2 ([4993077](https://github.com/chanzuckerberg/saber/commit/4993077afd1b2a1dcfc9efe5102aaee1d0690f3b))
* **main:** release saber-em 0.2.0 ([0c1858c](https://github.com/chanzuckerberg/saber/commit/0c1858c9d61f64a159e2da3a52c0e7e288db7edc))
* **main:** release saber-em 0.2.0 ([fcf3161](https://github.com/chanzuckerberg/saber/commit/fcf3161d5c2353c701ff460e81c53a7ac1a753f8))

## [0.7.2](https://github.com/chanzuckerberg/saber/compare/saber-v0.7.1...saber-v0.7.2) (2025-08-07)


### 🐞 Bug Fixes

* dynamic versioning ([cf280dd](https://github.com/chanzuckerberg/saber/commit/cf280dd2dca03eb645e5bd7fa6d1b5d554173cc4))
* dynamic versioning ([d1a67c8](https://github.com/chanzuckerberg/saber/commit/d1a67c808a05e606718624d587279260d12fe289))

## [0.7.1](https://github.com/chanzuckerberg/saber/compare/saber-v0.7.0...saber-v0.7.1) (2025-08-07)


### 🐞 Bug Fixes

* release-please ([10e89a6](https://github.com/chanzuckerberg/saber/commit/10e89a6a06850e29a3d196a421ae3c4d84f25324))
* release-please ([90052a3](https://github.com/chanzuckerberg/saber/commit/90052a3984eddad8314f35f5ec158fb54dbd621f))

## [0.7.0](https://github.com/chanzuckerberg/saber/compare/saber-v0.6.0...saber-v0.7.0) (2025-08-07)


### ✨ Features

* Image Saving and SAM2 Mask Batching ([df4ab57](https://github.com/chanzuckerberg/saber/commit/df4ab57d693af2451416ab5306ed596e936e5827))


### 🐞 Bug Fixes

* add README to PyPI description ([d9556fb](https://github.com/chanzuckerberg/saber/commit/d9556fb25453f2f15fbada3e1ccb51696d4aed4f))
* make sure inference state only read once ([f6963e7](https://github.com/chanzuckerberg/saber/commit/f6963e7545735ddbcfd46bd55048881e5072cc7a))
* make sure inference state only read once ([3672dce](https://github.com/chanzuckerberg/saber/commit/3672dce272d990010ce6f3c5d777e02e703d4ef7))


### 🧹 Miscellaneous Chores

* CCIE-4984 conform to open sourcing guidelines ([0badd9c](https://github.com/chanzuckerberg/saber/commit/0badd9c579bd2cf94f57d977c482343bb5e79206))
* CCIE-4984 conform to open sourcing guidelines ([34fdf98](https://github.com/chanzuckerberg/saber/commit/34fdf9885140778f57667672358458744b6581cd))

## [0.6.0](https://github.com/chanzuckerberg/saber/compare/saber-v0.5.0...saber-v0.6.0) (2025-07-10)


### ✨ Features

* add release-please configuration ([16da577](https://github.com/chanzuckerberg/saber/commit/16da5771c2ba345ba0b2ebd0ec6d7e5e63280da4))
* add release-please configuration ([6e8d558](https://github.com/chanzuckerberg/saber/commit/6e8d55899533ed950433fee339c951a9a31d59b9))
* prepare for v0.5.0 release ([6cafc31](https://github.com/chanzuckerberg/saber/commit/6cafc31c4f3a5f71b490779aaed795a0f394ef8e))


### 🐞 Bug Fixes

* make sure packages get submitted to pypi ([231361e](https://github.com/chanzuckerberg/saber/commit/231361eebab8a35b42e6f1466c5439a357f8b6df))
* make sure search pattern is correct ([2ffbb53](https://github.com/chanzuckerberg/saber/commit/2ffbb53c9aa64528095e331cd3203f5d151a1a78))
* rename release-please config file ([9403ffb](https://github.com/chanzuckerberg/saber/commit/9403ffbb3d293bc2b171828df81c8196ad977cb8))
* rename release-please config file ([03fe8c2](https://github.com/chanzuckerberg/saber/commit/03fe8c2f6df1b80b48d4ac696dbde85dc507eed0))
* resolve release please CI ([43a6847](https://github.com/chanzuckerberg/saber/commit/43a68470b266cfb9583be17d926549af7af7b09e))
* resolve release please CI ([dd26ff5](https://github.com/chanzuckerberg/saber/commit/dd26ff5230c636a61d5c34dff28d5fcfd554e1d0))
* use PAT for release-please to bypass org permissions ([c135a97](https://github.com/chanzuckerberg/saber/commit/c135a97bd76cfdcbf2a2edfa5ea38ed97bc3be3f))
* use PAT for release-please to bypass org permissions ([24b18d3](https://github.com/chanzuckerberg/saber/commit/24b18d30027315c9007d9db38b6b8637a3832a55))
* use the organization token ([b4b9d11](https://github.com/chanzuckerberg/saber/commit/b4b9d1179817c79b48af7b623fc1a9c82a9ca88d))


### 📝 Documentation

* better description for importing data to copick ([e5e9b64](https://github.com/chanzuckerberg/saber/commit/e5e9b64c514e9b8f5d7172d39dcb8f0e6b6d35c6))


### 🧹 Miscellaneous Chores

* **main:** release saber 0.3.0 ([0121f88](https://github.com/chanzuckerberg/saber/commit/0121f887643c6e78d7f619d86981c6bac4adda62))
* **main:** release saber 0.3.0 ([d94c1ed](https://github.com/chanzuckerberg/saber/commit/d94c1eda84bf0f14a001a2c2ee9d1017d8ee49b5))
* **main:** release saber-em 0.2.0 ([0c1858c](https://github.com/chanzuckerberg/saber/commit/0c1858c9d61f64a159e2da3a52c0e7e288db7edc))
* **main:** release saber-em 0.2.0 ([fcf3161](https://github.com/chanzuckerberg/saber/commit/fcf3161d5c2353c701ff460e81c53a7ac1a753f8))

## [0.3.0](https://github.com/chanzuckerberg/saber/compare/saber-v0.2.0...saber-v0.3.0) (2025-07-10)


### ✨ Features

* add release-please configuration ([16da577](https://github.com/chanzuckerberg/saber/commit/16da5771c2ba345ba0b2ebd0ec6d7e5e63280da4))
* add release-please configuration ([6e8d558](https://github.com/chanzuckerberg/saber/commit/6e8d55899533ed950433fee339c951a9a31d59b9))


### 🐞 Bug Fixes

* make sure packages get submitted to pypi ([231361e](https://github.com/chanzuckerberg/saber/commit/231361eebab8a35b42e6f1466c5439a357f8b6df))
* make sure search pattern is correct ([2ffbb53](https://github.com/chanzuckerberg/saber/commit/2ffbb53c9aa64528095e331cd3203f5d151a1a78))
* rename release-please config file ([9403ffb](https://github.com/chanzuckerberg/saber/commit/9403ffbb3d293bc2b171828df81c8196ad977cb8))
* rename release-please config file ([03fe8c2](https://github.com/chanzuckerberg/saber/commit/03fe8c2f6df1b80b48d4ac696dbde85dc507eed0))
* resolve release please CI ([43a6847](https://github.com/chanzuckerberg/saber/commit/43a68470b266cfb9583be17d926549af7af7b09e))
* resolve release please CI ([dd26ff5](https://github.com/chanzuckerberg/saber/commit/dd26ff5230c636a61d5c34dff28d5fcfd554e1d0))
* use PAT for release-please to bypass org permissions ([c135a97](https://github.com/chanzuckerberg/saber/commit/c135a97bd76cfdcbf2a2edfa5ea38ed97bc3be3f))
* use PAT for release-please to bypass org permissions ([24b18d3](https://github.com/chanzuckerberg/saber/commit/24b18d30027315c9007d9db38b6b8637a3832a55))
* use the organization token ([b4b9d11](https://github.com/chanzuckerberg/saber/commit/b4b9d1179817c79b48af7b623fc1a9c82a9ca88d))


### 📝 Documentation

* better description for importing data to copick ([e5e9b64](https://github.com/chanzuckerberg/saber/commit/e5e9b64c514e9b8f5d7172d39dcb8f0e6b6d35c6))


### 🧹 Miscellaneous Chores

* **main:** release saber-em 0.2.0 ([0c1858c](https://github.com/chanzuckerberg/saber/commit/0c1858c9d61f64a159e2da3a52c0e7e288db7edc))
* **main:** release saber-em 0.2.0 ([fcf3161](https://github.com/chanzuckerberg/saber/commit/fcf3161d5c2353c701ff460e81c53a7ac1a753f8))

## [0.2.0](https://github.com/chanzuckerberg/saber/compare/saber-em-v0.1.0...saber-em-v0.2.0) (2025-07-10)


### ✨ Features

* add release-please configuration ([16da577](https://github.com/chanzuckerberg/saber/commit/16da5771c2ba345ba0b2ebd0ec6d7e5e63280da4))
* add release-please configuration ([6e8d558](https://github.com/chanzuckerberg/saber/commit/6e8d55899533ed950433fee339c951a9a31d59b9))


### 🐞 Bug Fixes

* make sure search pattern is correct ([2ffbb53](https://github.com/chanzuckerberg/saber/commit/2ffbb53c9aa64528095e331cd3203f5d151a1a78))
* rename release-please config file ([9403ffb](https://github.com/chanzuckerberg/saber/commit/9403ffbb3d293bc2b171828df81c8196ad977cb8))
* rename release-please config file ([03fe8c2](https://github.com/chanzuckerberg/saber/commit/03fe8c2f6df1b80b48d4ac696dbde85dc507eed0))
* resolve release please CI ([43a6847](https://github.com/chanzuckerberg/saber/commit/43a68470b266cfb9583be17d926549af7af7b09e))
* resolve release please CI ([dd26ff5](https://github.com/chanzuckerberg/saber/commit/dd26ff5230c636a61d5c34dff28d5fcfd554e1d0))
* use PAT for release-please to bypass org permissions ([c135a97](https://github.com/chanzuckerberg/saber/commit/c135a97bd76cfdcbf2a2edfa5ea38ed97bc3be3f))
* use PAT for release-please to bypass org permissions ([24b18d3](https://github.com/chanzuckerberg/saber/commit/24b18d30027315c9007d9db38b6b8637a3832a55))
* use the organization token ([b4b9d11](https://github.com/chanzuckerberg/saber/commit/b4b9d1179817c79b48af7b623fc1a9c82a9ca88d))


### 📝 Documentation

* better description for importing data to copick ([e5e9b64](https://github.com/chanzuckerberg/saber/commit/e5e9b64c514e9b8f5d7172d39dcb8f0e6b6d35c6))
