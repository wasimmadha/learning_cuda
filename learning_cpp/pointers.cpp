#include <stdio.h>

int main(){
    int x = 10;
    int* ptr = &x;

    printf("Address of x: , %p\n", ptr);
    printf("Value of x: %d\n", *ptr);

    int arr[] = {10, 12, 13, 14, 15};

    // Array is a pointer to first element
    printf("Array: %p\n", arr);

    int* ptr1 = arr; // Pointer to the first element of array

    printf("Element at position one: %d\n", *ptr1);

    ptr1 += 1;

    printf("Element at position two: %d\n", *ptr1);

    // Writing it in a loop:

    ptr1 = arr;

    int length = sizeof(arr) / sizeof(arr[0]);
    
    for (int i = 0; i < length; i++){
        printf("%d ", *ptr1);
        printf("%p\n", ptr1);

        ptr1++;
    }

    return 0;
}