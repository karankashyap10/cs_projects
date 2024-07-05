#include<iostream>
#include<fstream>
#include<string>
using namespace std;

//function for creating a new file.
void newFile(const string& filename){
    ofstream file(filename);     
    if(file.is_open()){
        cout<<"New File named "<<filename<<" has been created. Open the file to start working in it. "<< endl;
        file.close();
    }
    else{
        cerr<<"Error"<< endl;
    }
    cout<<endl;
    cout<<endl;
}

//function for saving the file
void saveFile(const string& filename,const string& content){
    ofstream file(filename);
    if(file.is_open()){
        file<<content;
        file.close();
        cout<<"File "<<filename<<" saved"<< endl;
    }
    else{
        cerr<<"Error"<< endl;
    }
}

//function for writing in the opened file.
void writeFile(string& content, const string& filename) {
    cout << endl;
    size_t cursorPosition = content.length(); // Track cursor position
    // Print current content
    cout << content << endl;

    while (true) {      
        string input;
        getline(cin,input);

        if (input =="\n" ) {
            // Handle Enter key to add new line
            content.insert(cursorPosition," ");
            cursorPosition++;
        } else if (input == "\b" && cursorPosition > 0) {
            // Handle Backspace key
            content.erase(cursorPosition - 1, 1);
            cursorPosition--;
        } else if (input == "\x7F" && cursorPosition < content.length()) {
            // Handle Delete key
            content.erase(cursorPosition, 1);
        } else if (input == "s") {
            // Handle 'save' key for saving
            saveFile(filename, content);
        } else if (input == "e") {
            // Handle 'exit' key to quit
            break;
        } else {
            // Handle normal character input
            content.insert(cursorPosition,input);
            cursorPosition+=input.length();
            content.insert(cursorPosition," ");
            cursorPosition++;
        }
    }
}

//function for opening a existing file.
void openFile(const string&filename,string& content){
    ifstream file(filename);
    if(file.is_open()){
        string line;
        cout<<"File Content: \n"<< endl;
        while(getline(file,line)){
            cout<<line<< endl;
        }
        file.close();
        cout<<"Type 's' for saving the file and 'e' for exiting the file."<< endl;
        cout<<"Start Writing: "<< endl;
        cout<<endl;
        writeFile(content,filename);
    }
    else{
        cerr<<"Error"<< endl;
    }
}


int main(){
    string filename;
    string choice;
    string content;

    while(true){
        cout << "Text Editor Menu:\n"<< endl;
        cout << "1. Create new file\n";
        cout << "2. Open existing file\n";
        cout << "Enter your choice: ";
        cin>>choice;
        if (choice == "1") {
            cout << "Enter filename: ";
            cin>>filename;
            newFile(filename);
        }
        else if (choice == "2") {
            cout << "Enter filename: ";
            cin>>filename;
            openFile(filename,content);
        } 
    }
}