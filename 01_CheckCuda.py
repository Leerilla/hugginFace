# 토치가 설치되었는지 확인
import torch
x = torch.rand(5, 3)
print(x)

# torch 상태 확인하기
result = torch.cuda.is_available()
print(result)

# 만약 위코드가 실행되지 않는다면 torch가 설치되지 않은 것이다.
# 환경에서 torch가 체크되어 있는지 봐야 한다.
