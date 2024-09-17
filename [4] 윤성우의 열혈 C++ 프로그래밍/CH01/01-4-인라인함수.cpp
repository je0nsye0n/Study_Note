// 인라인 함수 : 프로그램 코드라인 안으로 들어가 버린 함수

#include<iostream>
#define SQUARE(x) ((x)*(x))

int main(void){
    std::cout<<SQUARE(4)<<std::endl;
    return 0;
}