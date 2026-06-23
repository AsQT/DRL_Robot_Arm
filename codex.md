Hãy kiểm tra toàn bộ thư mục:

`DRL_Pathplanning_trainning/`

Sau đó tạo file báo cáo:

`DRL_Pathplanning_trainning/struc.txt`

## Mục tiêu

File `struc.txt` phải mô tả chi tiết, có hệ thống và bám sát mã nguồn thực tế của thuật toán SAC đang được sử dụng để huấn luyện robot lập kế hoạch đường đi.

Không được chỉ trình bày lý thuyết SAC chung chung. Mọi thông số như số lớp, số neuron, activation, observation dimension, action dimension, reward coefficient, learning rate, buffer size... phải được truy vết trực tiếp từ mã nguồn, file cấu hình, checkpoint metadata hoặc tham số khởi tạo hiện tại.

Không được tự bịa thông số. Nếu một thông tin không tồn tại hoặc không thể xác định từ mã nguồn, phải ghi rõ:

`Không xác định được từ mã nguồn hiện tại`

và chỉ ra cần kiểm tra file hoặc biến nào để xác định.

---

# Nội dung bắt buộc của `struc.txt`

## 1. Tổng quan hệ thống huấn luyện

Mô tả:

* Mục tiêu của mô hình SAC.
* Robot hoặc môi trường đang được điều khiển.
* Không gian trạng thái.
* Không gian hành động.
* Điều kiện bắt đầu episode.
* Điều kiện kết thúc episode.
* Điều kiện thành công.
* Điều kiện thất bại.
* Luồng dữ liệu tổng quát:

```text
Environment
    -> raw observation
    -> observation preprocessing
    -> SAC Actor
    -> action
    -> action scaling/clipping
    -> environment step
    -> next observation, reward, terminated, truncated
    -> replay buffer
    -> Actor/Critic update
```

Phải ghi rõ các file và hàm tương ứng với từng bước.

---

## 2. Các file liên quan đến SAC

Liệt kê toàn bộ file có liên quan đến:

* Khởi tạo môi trường.
* Observation.
* Action.
* Reward.
* Collision checking.
* Goal generation.
* Obstacle generation.
* Huấn luyện.
* Đánh giá.
* Actor.
* Critic.
* Replay buffer.
* Normalization.
* Save/load model.
* Stable-Baselines3 hoặc framework RL đang sử dụng.
* Custom policy hoặc custom feature extractor nếu có.

Với mỗi file, mô tả:

```text
Tên file:
Vai trò:
Class chính:
Function chính:
Dữ liệu đầu vào:
Dữ liệu đầu ra:
```

---

## 3. Xác định framework và implementation SAC

Ghi rõ:

* SAC được tự cài đặt hay sử dụng thư viện.
* Nếu dùng Stable-Baselines3, ghi rõ class được dùng.
* Policy được dùng, ví dụ:

  * MlpPolicy
  * MultiInputPolicy
  * CnnPolicy
  * custom policy
* Phiên bản thư viện nếu có thể xác định.
* Device huấn luyện:

  * CPU
  * CUDA
  * auto
* Kiểu dữ liệu tensor.
* Seed.
* Số environment chạy song song.
* Có sử dụng `DummyVecEnv`, `SubprocVecEnv` hoặc vectorized environment hay không.
* Có sử dụng `VecNormalize` hay normalization tùy chỉnh hay không.

---

## 4. Cấu trúc toàn bộ các mạng neural trong SAC

Phải phân tích tất cả các mạng có trong implementation hiện tại, không chỉ Actor.

Tối thiểu cần kiểm tra:

1. Actor network.
2. Critic Q1 network.
3. Critic Q2 network.
4. Target Critic Q1.
5. Target Critic Q2.
6. Feature extractor nếu có.
7. Mạng hoặc biến dùng để học entropy coefficient nếu có.
8. Bất kỳ mạng phụ nào khác.

Với từng mạng, trình bày theo dạng bảng văn bản:

```text
Tên mạng:
Vai trò:
Input dimension:
Output dimension:

Layer 1:
- Loại layer:
- Input:
- Output:
- Số neuron:
- Activation:
- Có bias hay không:

Layer 2:
...

Output layer:
...
```

Phải chỉ ra chính xác:

* Số hidden layer.
* Số neuron mỗi hidden layer.
* Activation function.
* Input dimension.
* Output dimension.
* Có LayerNorm, BatchNorm, Dropout hay không.
* Cách khởi tạo weight nếu được cấu hình.
* Actor và Critic có dùng chung feature extractor hay không.
* Q1 và Q2 có dùng chung tham số hay là hai mạng độc lập.
* Target Critic được tạo và cập nhật như thế nào.
* Hệ số Polyak update `tau`.
* Tần suất cập nhật target network.

### Lưu ý về “kernel”

Nếu mạng hiện tại là MLP dùng `Linear/FullyConnected`, phải giải thích rõ:

* MLP không có convolution kernel như CNN.
* Mỗi layer dùng ma trận trọng số có kích thước:

```text
[out_features, in_features]
```

* Tổng số weight và bias của từng layer.
* Tổng số tham số của từng mạng.

Ví dụ cách trình bày:

```text
Linear(24 -> 256)
Weight matrix: [256, 24]
Bias vector: [256]
Số tham số: 256*24 + 256
```

Nếu có CNN, phải ghi rõ cho từng convolution layer:

* Kernel size.
* Stride.
* Padding.
* Số input channel.
* Số output channel.
* Kích thước feature map đầu vào và đầu ra.

Không được gọi nhầm ma trận trọng số của MLP là convolution kernel.

---

## 5. Actor network

Phân tích riêng Actor:

* Actor nhận observation nào.
* Observation có được flatten hay concatenate không.
* Kích thước tensor input.
* Các hidden layer.
* Actor tạo ra:

  * mean
  * log standard deviation
* Kích thước `mean`.
* Kích thước `log_std`.
* Giới hạn `log_std`.
* Cách tạo phân phối Gaussian.
* Cách lấy mẫu bằng reparameterization trick.
* Công thức:

```text
u = mean + std * epsilon
a = tanh(u)
```

* Có dùng `tanh` để giới hạn action không.
* Có hiệu chỉnh log probability sau `tanh` không.
* Action sau Actor nằm trong khoảng nào.
* Action được scale về giới hạn thật của môi trường như thế nào.
* Action biểu diễn:

  * delta joint position
  * joint velocity
  * absolute joint position
  * Cartesian displacement
  * hoặc dạng khác
* Đơn vị action.
* Action clipping.
* Hệ số scale của từng action dimension.

Phải đưa ra ví dụ cụ thể về shape:

```text
Observation batch: [batch_size, obs_dim]
Mean: [batch_size, action_dim]
Log std: [batch_size, action_dim]
Sampled action: [batch_size, action_dim]
```

---

## 6. Critic network

Phân tích riêng Critic:

* Critic nhận observation và action như thế nào.
* Observation và action được concatenate tại đâu.
* Input dimension:

```text
critic_input_dim = observation_dim + action_dim
```

* Output của mỗi Critic.
* Q1 và Q2 có kiến trúc giống nhau không.
* Hai Critic có độc lập hoàn toàn không.
* Vai trò của Double Q-learning.
* Cách lấy:

```text
min(Q1, Q2)
```

* Target Q-value được tính như thế nào.
* Có trừ entropy term hay không.
* Discount factor `gamma`.
* Target update coefficient `tau`.

Trình bày công thức đúng với implementation thực tế:

```text
target_q = reward + gamma * (1 - done) *
           (min(target_q1, target_q2) - alpha * next_log_prob)
```

Nếu implementation xử lý `terminated` và `truncated` khác nhau, phải mô tả chính xác.

---

## 7. Entropy coefficient

Mô tả:

* `alpha` được đặt cố định hay tự động học.
* Giá trị ban đầu.
* `ent_coef`.
* `target_entropy`.
* Nếu dùng `auto`, trình bày:

  * biến `log_alpha`
  * optimizer của alpha
  * learning rate
  * loss dùng để cập nhật alpha
* Quan hệ giữa entropy và khả năng exploration.

Không được chỉ ghi lý thuyết; phải chỉ ra giá trị và cấu hình trong mã nguồn hiện tại.

---

## 8. Cấu trúc observation

Liệt kê toàn bộ các thành phần observation đúng thứ tự được ghép vào vector.

Với từng thành phần, ghi:

```text
Tên thành phần:
Nguồn dữ liệu:
File và function tạo dữ liệu:
Số chiều:
Đơn vị gốc:
Khoảng giá trị vật lý:
Cách chuẩn hóa:
Khoảng sau chuẩn hóa:
Ý nghĩa đối với quá trình tránh vật cản:
```

Cần kiểm tra đầy đủ các thành phần có thể tồn tại như:

* Joint position hiện tại.
* Joint velocity.
* TCP position.
* TCP orientation.
* Goal position.
* Goal orientation.
* Vector từ TCP đến goal.
* Khoảng cách đến goal.
* Obstacle position.
* Obstacle size.
* Khoảng cách từ TCP đến obstacle.
* Khoảng cách từ từng link đến obstacle.
* Collision flag.
* Previous action.
* Step progress.
* Các đặc trưng khác.

Phải ghi chính xác thứ tự concatenate, ví dụ:

```text
observation = [
    joint_positions,
    tcp_position,
    tcp_orientation,
    goal_position,
    obstacle_position,
    obstacle_size,
    ...
]
```

Sau đó lập bảng index:

```text
Index       Thành phần                 Số chiều
0-5         Joint position             6
6-8         TCP position               3
...
Tổng observation dimension: ...
```

Nếu observation là `Dict`, phải trình bày từng key, shape và policy xử lý tương ứng.

---

## 9. Chuẩn hóa dữ liệu observation

Phân tích từ dữ liệu môi trường thực tế đến tensor đưa vào Actor:

```text
Raw environment data
-> unit conversion
-> clipping
-> min-max normalization hoặc standardization
-> concatenate
-> numpy array
-> vectorized environment
-> PyTorch tensor
-> neural network
```

Phải kiểm tra và ghi rõ:

* Có đổi degree sang radian hay không.
* Có chuẩn hóa joint position theo joint limit hay không.
* Có chuẩn hóa Cartesian position theo workspace limit hay không.
* Có chuẩn hóa quaternion hay không.
* Có chuẩn hóa obstacle position và size hay không.
* Có clipping hay không.
* Có dùng running mean và variance hay không.
* Có dùng `VecNormalize` hay không.
* Có lưu normalization statistics khi save model hay không.
* Khi inference có load lại normalization statistics hay không.

Đối với mỗi thành phần, ghi công thức thật đang dùng.

Ví dụ:

```text
x_norm = 2 * (x - x_min) / (x_max - x_min) - 1
```

hoặc:

```text
x_norm = (x - mean) / sqrt(var + epsilon)
```

Nếu hiện tại dữ liệu chưa được chuẩn hóa, phải ghi rõ:

```text
Thành phần này hiện đang được đưa trực tiếp vào mạng, chưa chuẩn hóa.
```

Sau đó đánh giá rủi ro:

* Các thành phần có scale quá khác nhau không.
* Có thể làm gradient mất cân bằng không.
* Có ảnh hưởng đến convergence không.

---

## 10. Cấu trúc action

Mô tả đầy đủ:

* Action dimension.
* Ý nghĩa từng phần tử.
* Khoảng action space khai báo trong Gymnasium.
* Dữ liệu Actor xuất ra.
* Cách unscale action.
* Hệ số action scale.
* Giới hạn joint.
* Giới hạn velocity.
* Cách xử lý action vượt giới hạn.
* Action có được cộng dồn vào joint hiện tại không.
* Action được gửi tới:

  * động học thuận
  * động học nghịch
  * MoveIt
  * mock hardware
  * Gazebo
  * hoặc mô hình robot nội bộ

Lập bảng:

```text
Action index   Ý nghĩa   Khoảng RL   Hệ số scale   Đơn vị vật lý
0              Δq1       [-1,1]      ...           rad
...
```

---

## 11. Hàm phần thưởng

Phân tích trực tiếp hàm reward hiện tại.

Không được chỉ mô tả bằng câu. Phải khôi phục thành công thức tổng quát:

```text
R_total =
    w_goal * R_goal
  + w_progress * R_progress
  + w_collision * R_collision
  + w_action * R_action
  + w_smooth * R_smooth
  + w_success * R_success
  + w_timeout * R_timeout
  + ...
```

Với từng thành phần reward, ghi:

```text
Tên:
Công thức:
Hệ số:
Điều kiện áp dụng:
Khoảng giá trị:
Ý nghĩa:
File và dòng code:
```

Tối thiểu cần kiểm tra:

* Distance-to-goal reward.
* Progress reward.
* Success bonus.
* Collision penalty.
* Obstacle proximity penalty.
* Joint limit penalty.
* Action magnitude penalty.
* Action smoothness penalty.
* Time-step penalty.
* Timeout penalty.
* Orientation error reward/penalty.
* Path length penalty.
* End-effector velocity penalty.
* Self-collision penalty nếu có.

Phải phân biệt:

* Reward dựa trên khoảng cách tuyệt đối.
* Reward dựa trên độ giảm khoảng cách giữa hai step:

```text
progress = previous_distance - current_distance
```

Ghi rõ hệ số số học thực tế, ví dụ:

```text
collision_penalty = -100.0
success_bonus = +200.0
distance_weight = -2.0
```

Nếu hệ số nằm trong file cấu hình, ghi rõ tên tham số.

---

## 12. Điều kiện terminated và truncated

Phân tích riêng:

### Terminated

* Goal reached.
* Collision.
* Invalid state.
* Joint limit violation.
* NaN.
* Các điều kiện kết thúc mang tính trạng thái cuối.

### Truncated

* Vượt quá maximum episode steps.
* Timeout.
* Điều kiện cắt episode khác.

Ghi rõ code kiểm tra, thứ tự ưu tiên và ảnh hưởng đến target Q-value.

---

## 13. Replay buffer

Mô tả:

* Buffer size.
* Kiểu replay buffer.
* Dữ liệu được lưu:

  * observation
  * action
  * reward
  * next observation
  * done
  * timeout
* Batch size.
* Learning starts.
* Sampling method.
* Có prioritized replay hay không.
* Có HER hay không.
* Có lưu normalization data hay không.
* Tần suất cập nhật mạng so với số environment step.

Ghi shape của một batch mẫu.

---

## 14. Hyperparameter SAC

Lập bảng toàn bộ hyperparameter tìm được:

```text
Tên tham số              Giá trị       Nguồn
learning_rate            ...
buffer_size              ...
learning_starts          ...
batch_size               ...
tau                      ...
gamma                    ...
train_freq               ...
gradient_steps           ...
action_noise             ...
replay_buffer_class      ...
optimize_memory_usage    ...
ent_coef                 ...
target_update_interval   ...
target_entropy            ...
use_sde                  ...
sde_sample_freq          ...
policy_kwargs             ...
seed                     ...
device                   ...
```

Nếu learning rate sử dụng schedule, phải mô tả schedule đó.

---

## 15. Optimizer và loss function

Mô tả riêng:

### Critic loss

```text
critic_loss =
MSE(Q1(current_obs, action), target_q)
+
MSE(Q2(current_obs, action), target_q)
```

### Actor loss

```text
actor_loss =
mean(alpha * log_prob - min(Q1, Q2))
```

### Entropy coefficient loss

Mô tả đúng với implementation hiện tại.

Ghi rõ:

* Optimizer.
* Learning rate.
* Epsilon.
* Weight decay.
* Gradient clipping nếu có.
* Thứ tự cập nhật Actor, Critic và alpha.
* Có freeze Critic khi update Actor hay không.

---

## 16. Quá trình huấn luyện

Mô tả theo từng bước:

1. Reset environment.
2. Sinh start state.
3. Sinh goal.
4. Sinh obstacle.
5. Tạo observation.
6. Actor lấy action.
7. Environment thực hiện action.
8. Tính next observation.
9. Tính reward.
10. Kiểm tra collision.
11. Kiểm tra success.
12. Lưu transition vào replay buffer.
13. Sample mini-batch.
14. Update Critic.
15. Update Actor.
16. Update entropy coefficient.
17. Update target networks.
18. Ghi log.
19. Save checkpoint.
20. Evaluate model.

Ghi rõ mỗi bước nằm trong thư viện SAC hay custom environment.

---

## 17. Logging, checkpoint và evaluation

Kiểm tra:

* TensorBoard log.
* CSV log.
* Episode reward.
* Episode length.
* Success rate.
* Collision rate.
* Mean distance.
* Actor loss.
* Critic loss.
* Entropy coefficient.
* Learning rate.
* Save frequency.
* Checkpoint path.
* Best model path.
* Evaluation frequency.
* Số evaluation episode.
* Deterministic action khi evaluation.
* Cách load model để inference.

Ghi rõ file model và file normalization cần thiết.

---

## 18. Số lượng tham số mạng

Tính số tham số cho:

* Actor feature network.
* Actor mean head.
* Actor log_std head.
* Q1.
* Q2.
* Target Q1.
* Target Q2.
* Tổng số trainable parameters.
* Tổng số parameters kể cả target networks.

Có thể sử dụng script Python hoặc trực tiếp load model để đếm:

```python
sum(p.numel() for p in model.parameters())
```

Phân biệt:

* Trainable parameters.
* Non-trainable parameters.
* Target network parameters.

Nếu dùng Stable-Baselines3, hãy load hoặc khởi tạo model với đúng environment để in trực tiếp cấu trúc:

```python
print(model.policy)
print(model.actor)
print(model.critic)
print(model.critic_target)
```

Đưa output đã rút gọn, dễ đọc vào `struc.txt`.

---

## 19. Shape tracing

Thêm một mục theo dõi shape qua toàn bộ mạng.

Ví dụ:

```text
Raw observation:             (obs_dim,)
Vectorized observation:      (n_envs, obs_dim)
Replay buffer batch:         (batch_size, obs_dim)
Actor hidden layer 1:        (batch_size, 256)
Actor hidden layer 2:        (batch_size, 256)
Actor mean:                  (batch_size, action_dim)
Actor log_std:               (batch_size, action_dim)
Sampled action:              (batch_size, action_dim)

Critic observation:          (batch_size, obs_dim)
Critic action:               (batch_size, action_dim)
Critic concatenated input:   (batch_size, obs_dim + action_dim)
Q1 output:                   (batch_size, 1)
Q2 output:                   (batch_size, 1)
```

Các dimension phải lấy đúng từ project hiện tại.

---

## 20. Phân tích một transition thực tế

Chạy một lần `reset()` và tối thiểu một lần `step()` trong môi trường, nếu môi trường có thể chạy độc lập an toàn.

Ghi một ví dụ thực tế:

```text
Observation shape:
Observation dtype:
Observation min:
Observation max:
Observation mean:

Action:
Scaled physical action:

Reward components:
Total reward:

Terminated:
Truncated:
Info:
```

Không cần chạy huấn luyện dài. Không được làm hỏng checkpoint hiện có.

Nếu môi trường không thể chạy do thiếu ROS, Gazebo hoặc dependency, ghi rõ lỗi và vẫn tiếp tục phân tích tĩnh mã nguồn.

---

## 21. Các vấn đề và đề xuất cải thiện

Sau phần mô tả đúng implementation hiện tại, thêm phần đánh giá kỹ thuật.

Phải kiểm tra và nhận xét:

* Observation có thiếu thông tin vật cản không.
* Vị trí và kích thước vật cản có được đưa vào observation không.
* Observation có Markov đầy đủ không.
* Các thành phần observation có scale đồng đều không.
* Quaternion có được normalize không.
* Joint angle có dùng biểu diễn `sin/cos` hay chỉ dùng góc trực tiếp.
* Reward có quá thưa không.
* Success bonus có lấn át các reward khác không.
* Collision penalty có đủ lớn không.
* Obstacle proximity penalty có gây local optimum không.
* Action scale có quá lớn làm robot xuyên vật cản không.
* Action scale có quá nhỏ làm episode timeout không.
* Có penalty cho thay đổi action đột ngột không.
* Có kiểm tra toàn bộ link robot hay chỉ TCP.
* Có nguy cơ Actor chỉ học đi thẳng tới goal mà bỏ qua obstacle không.
* Goal và obstacle randomization có đủ đa dạng không.
* Replay buffer có chứa đủ transition collision và near-collision không.
* Có curriculum learning không.
* Có cần HER không.
* Có cần observation normalization không.
* Có cần frame stacking hoặc previous action không.
* Có cần thêm khoảng cách từ từng link tới obstacle không.
* Có cần thêm vector goal tương đối thay vì chỉ dùng tọa độ tuyệt đối không.

Mỗi đề xuất phải phân biệt rõ:

```text
Hiện trạng:
Rủi ro:
Đề xuất:
Mức ưu tiên: Cao / Trung bình / Thấp
```

Không được tự sửa thuật toán trong nhiệm vụ này, trừ khi việc sửa nhỏ là cần thiết để chạy script phân tích và không làm thay đổi logic huấn luyện.

---

## 22. Sơ đồ kiến trúc SAC dạng ASCII

Thêm sơ đồ dễ hiểu, ví dụ:

```text
                         +----------------------+
Observation ------------>| Actor feature MLP    |
                         +----------+-----------+
                                    |
                         +----------+-----------+
                         |                      |
                    Mean head              Log-Std head
                         |                      |
                         +----------+-----------+
                                    |
                          Gaussian Sampling
                                    |
                                  tanh
                                    |
                             Scaled Action
                                    |
                                    v
                              Environment
                                    |
              +---------------------+---------------------+
              |                                           |
              v                                           v
        Next Observation                              Reward/Done
              |                                           |
              +---------------- Replay Buffer ------------+
                                    |
                              Sample Batch
                                    |
                 +------------------+------------------+
                 |                                     |
                 v                                     v
          Q1(obs, action)                       Q2(obs, action)
                 |                                     |
                 +------------------+------------------+
                                    |
                               min(Q1, Q2)
                                    |
                      Actor/Critic/Alpha Update
```

Điều chỉnh sơ đồ theo đúng implementation thực tế.

---

## 23. Phân biệt training và inference

Lập bảng so sánh:

```text
Thành phần               Training                 Inference
Action sampling          stochastic               deterministic hoặc stochastic
Replay buffer            có                       không
Critic update             có                       không
Actor update              có                       không
Normalization             update statistics        frozen statistics
Obstacle randomization    ...                      ...
Goal randomization        ...                      ...
```

---

## 24. Nguồn truy vết

Cuối mỗi mục quan trọng phải chỉ ra nguồn trong project theo dạng:

```text
Nguồn:
- file: ...
- class: ...
- function: ...
- biến hoặc tham số: ...
```

Nếu có thể xác định số dòng ổn định, ghi thêm số dòng. Tuy nhiên không được chỉ phụ thuộc vào số dòng vì source code có thể thay đổi.

---

# Yêu cầu cách trình bày

* File phải là plain text UTF-8.
* Viết bằng tiếng Việt.
* Có tiêu đề và đánh số mục rõ ràng.
* Không sử dụng mô tả mơ hồ.
* Không được sao chép nguyên lý SAC chung mà không liên hệ với code hiện tại.
* Mọi con số phải có nguồn từ code.
* Công thức phải dễ đọc trong file `.txt`.
* Phân biệt rõ:

  * thông số đọc được từ code;
  * kết quả đo khi runtime;
  * nhận xét;
  * đề xuất.
* Không thay đổi logic huấn luyện hiện tại.
* Không xóa hoặc ghi đè model/checkpoint.
* Không chạy huấn luyện dài.
* Có thể tạo script phân tích tạm thời, nhưng phải xóa script tạm sau khi hoàn thành nếu không cần lưu lại.

---

# Kiểm tra sau khi tạo

Sau khi hoàn thành:

1. Xác nhận file tồn tại:

```bash
test -f DRL_Pathplanning_trainning/struc.txt
```

2. Kiểm tra file không rỗng:

```bash
test -s DRL_Pathplanning_trainning/struc.txt
```

3. Kiểm tra file có các mục chính:

```bash
grep -n "Actor" DRL_Pathplanning_trainning/struc.txt
grep -n "Critic" DRL_Pathplanning_trainning/struc.txt
grep -n "Observation" DRL_Pathplanning_trainning/struc.txt
grep -n "Reward" DRL_Pathplanning_trainning/struc.txt
grep -n "Replay buffer" DRL_Pathplanning_trainning/struc.txt
grep -n "Chuẩn hóa" DRL_Pathplanning_trainning/struc.txt
```

4. In ra:

   * đường dẫn file;
   * số dòng;
   * dung lượng file;
   * danh sách file mã nguồn đã phân tích.

5. Tóm tắt ngắn trong phản hồi:

   * observation dimension;
   * action dimension;
   * số lớp Actor;
   * số lớp Critic;
   * tổng số tham số;
   * reward chính;
   * normalization đang sử dụng;
   * các vấn đề kỹ thuật quan trọng nhất phát hiện được.

Hãy trực tiếp thực hiện việc đọc mã nguồn, phân tích, tạo file và kiểm tra file. Không chỉ trả lời hướng dẫn.
