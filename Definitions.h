#include <pthread.h>
#include <unistd.h>
#include <vector>
#include <fcntl.h>
#include <stdlib.h>
#include <sys/ipc.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <semaphore.h>
#include <fstream>
#include <sstream>
#include <cassert>
#include <iostream>

using namespace std;

#define CONFIGFILE "config.txt"
#define ITERATIONS 3
#define CONTEXTSWITCH sleep(0.1)
#define SIGNALSIZE 1

// Menu Operations
#define EXIT '1'
#define FRONTPROP '2'
#define BACKPROP '3'
#define DISPLAY '4'

// Signal Values for Pipes
#define WFRONTPROP "2"
#define WBACKPROP "3"
#define WDISPLAY "4"
#define WEXIT "1"

vector<int> topology;
vector<double> inputs;
vector<vector<vector<double>>> weights;

class Neuron;
typedef vector<Neuron> Layer;

class Neuron
{
public:
    Neuron(){};
    Neuron(unsigned myIndex, double val = 0);
    void setOutputVal(double val) { m_outputVal = val; }
    const double getOutputVal(void) const { return m_outputVal; }
    double &getOutputVal(void) { return m_outputVal; }
    const unsigned getIndex() const {return m_myIndex;}
private:
    double m_outputVal;
    unsigned m_myIndex;
};
struct Thread
{
    int writeToNext;
    Neuron* neuron;
    const Layer *threadlayer;
    const vector<vector<double>> *weights;

    Thread()
    {
        writeToNext = 4;
        neuron = nullptr;
        threadlayer = nullptr;
        weights = nullptr;
    }
    Thread(const int fd, const Layer *l, const vector<vector<double>> *w, Neuron* n)
        : writeToNext(fd), threadlayer(l), weights(w), neuron(n) {}
    void copy(const unsigned fd, const Layer *l, const vector<vector<double>> *w, Neuron* n)
    {
        writeToNext = fd;
        neuron = n;
        threadlayer = l;
        weights = w;
    }
};
struct NeuronArgs
{
    int writeToNext;
    double *inputVal;
    Neuron *neuron;

    NeuronArgs(int fd = 4, double *i = nullptr, Neuron*n = nullptr)
    {
        writeToNext = fd;
        neuron = n;
        inputVal = i;
    }
    void copy(const int fd, double *i, Neuron *n)
    {
        writeToNext = fd;
        inputVal = i;
        neuron = n;
    }
};

double backPropFunction1(double x)
{
    return ((x * x) + x + 1) / 2;
}

double backPropFunction2(double x)
{
    return ((x * x) - x) / 2;
}

Neuron::Neuron(unsigned myIndex, double val)
{
    m_outputVal = val;
    m_myIndex = myIndex;
}
void display(vector<Neuron> &layer, int t)
{
    cout << getpid() << " " << " " << t << " : ";
    for (int i = 0; i < layer.size(); i++)
        cout << layer[i].getOutputVal() << " ";
    cout << endl;
}
void close(const int fd1, const int fd2, const int fd3, const int fd4)
{
    close(fd1);
    close(fd2);
    close(fd3);
    close(fd4);
}
void close(const int fd1, const int fd2)
{
    close(fd1);
    close(fd2);
}
void signal_lock(const int lastRead)
{
    char temp;
    while (!read(lastRead, &temp, 1))
        CONTEXTSWITCH;
}
void signal_unlock(const int firstWrite)
{
    write(firstWrite, "1", 1);
}
class Network
{
    int lastRead, lastWrite;
    // IPC Pipe File Descriptors
    int nextRead, nextWrite;
    int prevRead, prevWrite;
    int signalNextRead;
    int signalNextWrite;
    int signalPrevRead;
    int signalPrevWrite;
    int id;
    // Current Layer of Process
    vector<Neuron> currlayer;

public:
    Network(const vector<int> topology);
    Network(const vector<vector<vector<double>>> topology);
    
    void InitLayer(Layer &currlayer, const vector<int> &topology, const int id);
    void LayersCreate(const vector<int> &, const int, const int,
                      const int, const int, const int, const int, const int);

    void LayerMain(const vector<int> &);
    void LastMain(const vector<int> &);

    void feedForward(vector<double> &inputVals);
    void feedForwardHidden();
    void feedBackward(vector<double>&resultVals);
    void feedBackwardHidden();
    void layerProcessing(const Layer &, const int);

    void Print();
    void End();
};

void display(vector<vector<vector<double>>> &vec)
{
    for (int i = 0; i < vec.size(); i++, cout << "\n\n")
        for (int j = 0; j < vec[i].size(); j++, cout << "\n")
            for (int k = 0; k < vec[i][j].size(); k++)
                cout << vec[i][j][k] << " ";
    cout << endl;
}
void display(vector<vector<double>> &vec)
{
    for (int i = 0; i < vec.size(); i++, cout << endl)
        for (int j = 0; j < vec[i].size(); j++)
            cout << vec[i][j] << " ";
    cout << endl;
}
void display(vector<double> &vec)
{
    for (int i = 0; i < vec.size(); i++)
        cout << vec[i] << " ";
    cout << endl;
}
void display(vector<int> &vec)
{
    for (int i = 0; i < vec.size(); i++)
        cout << vec[i] << " ";
    cout << endl;
}

void ReadFile()
{
    fstream file(CONFIGFILE, ios::in);
    string line, data;

    int tableNum = -1, rowNum = 0;
    while (getline(file, line))
    {
        stringstream stream(line);
        if (line[0] == '\r')
            weights.push_back(vector<vector<double>>()), tableNum++, rowNum = 0;
        else if (line[0] == '_')
        {
            getline(file, line);
            stringstream stream(line);
            while (getline(stream, data, ','))
            {
                inputs.push_back(stod(data));
                while (stream.peek() == ' ')
                {
                    int pos = stream.tellg();
                    stream.seekg(pos + 1);
                }
            }
            break;  
        }
        else
        {
            weights[tableNum].push_back(vector<double>());
            while (getline(stream, data, ','))
            {
                weights[tableNum][rowNum].push_back(stod(data));
                while (stream.peek() == ' ')
                {
                    int pos = stream.tellg();
                    stream.seekg(pos + 1);
                }
            }
            rowNum++;
        }
    }
    //weights.pop_back();
    //display(weights);

    topology.reserve(weights.size() + 1);
    for (int i = 0; i < weights.size(); i++)
        topology.emplace_back(weights[i].size());
    topology.emplace_back(1); 

    display(topology);
    //display(inputs);
}
