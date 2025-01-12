#include <iostream>
#include <string>
#include <cmath>

using namespace std;

string example_string(int x, int y) {
  if (x > y) {
    return "x is greater than y";
  }
  else{
    return "y is greater than x";
  }

}

int main() {
  // This is just for test

  cout << "Hello World!" << endl;
  cout << "Have a good day!" << endl;
  cout << 3 + 6 << endl;

  // multiple test at a time
  // this is also comment 
  
  int myNuber = 10;
  double flotNumber = 0.542; 
  char myletter = 'D';
  string str = "This is Wasim";
  bool myBoolean = true;

  const int floatNumber = 10;

  cout << str << endl;

  // String
  string firstName = "John ";
  string lastName = "Doe";
  string fullName = firstName + lastName;
  cout << fullName << endl;

  string txt = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
  cout << "The length of the txt string is: " << txt.length() << endl;


  cout << sqrt(64) << endl;
  cout << round(2.6) << endl;
  cout << log(2) << endl;

  string exampl = example_string(1000, 100);

  cout << exampl << endl;

  int day = 4;
  switch (day) {
    case 1:
      cout << "Monday" << endl;
      break;
    case 2:
      cout << "Tuesday" << endl;
      break;
    case 3:
      cout << "Wednesday" << endl;
      break;
    case 4:
      cout << "Thursday" << endl;
      break;
    case 5:
      cout << "Friday" << endl;
      break;
    case 6:
      cout << "Saturday" << endl;
      break;
    case 7:
      cout << "Sunday" << endl;
      break;
  }

  int i = 0;
  while (i < 5) {
    cout << "While: " << i << "\n";
    i++;
  }

  for (int i = 0; i < 5; i++) {
    cout <<  "For: " << i << "\n";
  }

  for (int i = 0; i < 10; i++) {
    if (i == 4) {
      break;
    }
    cout << "For Break: " << i << "\n";
  }

  string cars[5] = {"Volvo", "BMW", "Ford", "Mazda", "Tesla"};

  // Loop through strings
  for (int i = 0; i < 5; i++) {
    cout << cars[i] << "\n";
  }

  return 0;
}

