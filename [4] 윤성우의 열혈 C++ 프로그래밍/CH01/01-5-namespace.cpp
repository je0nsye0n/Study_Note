// namespace : c++로 넘어오면서 굉장히 방대한 양의 라이브러리와 식별자들이 나오게 되고 중복이 되는 상황이 발생하게 되면 구분을 할 수 없는 경우가 생긴다. 이를 해결하기 위해서 나온 개념이다.

#include<iostream>

/*
namespace Kim{
void func(){
        std::cout << "Kim Hello \n";
    }
}

namespace Lee{
void func(){
        std::cout << "Lee Hello \n";
    }
}

int main(void){
    Kim::func();
    Lee::func();
    return 0;
}

*/

// using 

using namespace std; // namespce std에 포함된 모든 것을 사용하겠다 == 특정 식별자인 std를 생략할 것이다

int main(void){
    cout << "Hello World\n"; 
    return 0;
}