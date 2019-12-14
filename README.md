# 2019.Fall.PatternRecognition.TermProject
## Implement the performance of 'Beyond Bags of Features: Spatial Pyramid Matching for Recognizing Natural Scene Categories'

## Code description

### Data Load
```
! kaggle competitions download -c 2019-ml-finalproject
os.mkdir('/content/input2')
zip_ref = zipfile.ZipFile("/content/2019-ml-finalproject.zip", 'r')
zip_ref.extractall("/content/input2")
zip_ref.close()
```

### Extract descriptors from train & test dataset
```
data_root_train = "/content/input2/train/"

train_images = []
labels = []
des_list = []
            
sift = cv2.xfeatures2d.SIFT_create()
for i in tqdm_notebook(os.listdir(data_root_train)):
  img_cls_path = data_root_train + i + "/"  # 이미지 클래스 path
  img_path = [img_cls_path + j for j in os.listdir(img_cls_path)] 
  # 각 클래스 내의 학습이미지 path

  if i == "BACKGROUND_Google":
    label = 0
  else:
    label = (df_train.index[df_train[1]==i] + 1).tolist()[0]

  for img in img_path:
    image = cv2.imread(img)
    image = cv2.resize(image, (256,256))
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    train_images.append(image_gray)

    kp, des = dense_sift_each(image_gray)
    des_list.append(des)
    labels.append(label)
    
des_tot = np.vstack((descriptor for descriptor in des_list))
```
```
data_root_test = "/content/input2/testAll_v2/"

img_list = os.listdir(data_root_test)
des_list_test = []
test_images = []

img_path_test = [data_root_test + i for i in img_list]

for img in tqdm_notebook(img_path_test):
  image = cv2.imread(img)
  image = cv2.resize(image, (256,256))
  image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  test_images.append(image_gray)

  kp, des = dense_sift_each(image_gray)
  des_list_test.append(des)
```

### Make codebook using KMeans clustering
```
codebooksize = 800
seeding = kmc2.kmc2(np.array(des_tot).reshape(-1,128), codebooksize)
Kmeas = MiniBatchKMeans(codebooksize, init=seeding).fit(np.array(des_tot).reshape(-1,128))
codeBook = Kmeas.cluster_centers_
```

### Spatial Pyramid Matching(SPM)
```
def build_spatial_pyramid(image, descriptor, level):
    step_size = 8
    height = int(image.shape[0] / step_size)    # 256/8 = 32 : des 개수
    width = int(image.shape[1] / step_size)

    # descriptor의 인덱스를 표시하기 위함(0~1023을 32*32의 배열로 표현)
    idx_crop = np.array(range(len(descriptor))).reshape(height,width)
    size = idx_crop.itemsize  # 배열의 각 요소의 바이트 사이즈 -> strides
    bh, bw = 2**(5-level), 2**(5-level)
    shape = (int(height/bh), int(width/bw), bh, bw) # level에 맞춰 이미지를 분할
    strides = size * np.array([width*bh, bw, width, 1])
    # 영상(여기서는 descriptor) 분할
    crops = np.lib.stride_tricks.as_strided(idx_crop, shape=shape, strides=strides)
    # 각각의 분할된 이미지에서 descriptor index를 1차원으로 만듦
    des_idxs = [col_block.flatten().tolist() for row_block in crops for col_block in row_block]

    # 위에 인덱스를 이용하여 
    pyramid = []
    for idxs in des_idxs:
        pyramid.append(np.asarray([descriptor[idx] for idx in idxs]))   
    return pyramid
```
```
def spatial_pyramid_matching(image, descriptor, codebook, level):
    pyramid = []
    pyramid += build_spatial_pyramid(image, descriptor, level=0)
    pyramid += build_spatial_pyramid(image, descriptor, level=1)
    pyramid += build_spatial_pyramid(image, descriptor, level=2)

    code = [input_vector_encoder(crop, codebook) for crop in pyramid]
    code_level_0 = 0.25 * np.asarray(code[0]).flatten()
    code_level_1 = 0.25 * np.asarray(code[1:5]).flatten()
    code_level_2 = 0.5 * np.asarray(code[5:]).flatten()
    
    return np.concatenate((code_level_0, code_level_1, code_level_2))
```

### SPM Kernel
```
def histogramIntersection(M, N):
    m = M.shape[0]
    n = N.shape[0]
    result = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            temp = np.sum(np.minimum(M[i], N[j]))
            result[i][j] = temp
    return result
```

### Process train & test data using each descriptors
```
X = [spatial_pyramid_matching(train_images[i], des_list[i], codeBook, level=2) for i in range(len(train_images))]

scaler = StandardScaler()
X = np.array(X)
X = scaler.fit_transform(X)
y = np.array(labels)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, random_state=0)
gramMatrix = histogramIntersection(X_train, X_train)
```
```
X_test = [spatial_pyramid_matching(test_images[i], des_list_test[i], codeBook, level=2) for i in range(len(test_images))]

X_test = np.array(X_test)
X_test = scaler.fit_transform(X_test)
predictMatrix = histogramIntersection(X_test, X_train)
```

### Train & predict
```
param_grid = {'kernel':['precomputed'], 'C':[0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]}
svc = SVC(gamma='scale')
grid = GridSearchCV(svc, param_grid, cv=5, n_jobs=-2)

%time grid.fit(gramMatrix, y_train)
print(grid.best_params_)

model = grid.best_estimator_
y_pred = model.predict(predictMatrix)
```

***
## Result
| Level | codebooksize | imagesize | spm | spm_kernel | scaler | accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 16 | default | X | X | X | 0.15602 |
| 0 | 200 | default | X | X | X | 0.41430 |
| 0 | 400 | default | X | X | X | 0.43617 |
| 2 | 200 | 256*256 | O | X | X | 0.49231 |
| 2 | 200 | 256*256 | O | O | X | 0.52718 |
| 2 | 800 | 256*256 | O | O | X | 0.57387 |
| 2 | 800 | 256*256 | O | O | O | 0.60756 |

***
## References
- [Beyond bags of features spatial pyramid matching for recognizing natural scene categories, CVPR 2006](https://inc.ucsd.edu/~marni/Igert/Lazebnik_06.pdf)
- [BoW, SPM](https://github.com/CyrusChiu/Image-recognition)
