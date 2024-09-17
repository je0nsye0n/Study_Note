#include <iostream>

int MyFunc(int a=0, int b=1){ // 이러한 식으로 인자가 전달이 안되었을 때, 작동이 되도록 하는 디폴드 값이 설정 가능하다.
    return a+b;
}

int main(void){
    std::cout<<MyFunc(1);
    return 0;
}