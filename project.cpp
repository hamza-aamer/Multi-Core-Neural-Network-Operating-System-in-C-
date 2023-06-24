#include "Definitions.h"

void Network::InitLayer(Layer &currlayer, const vector<int> &topology, const int id)
{
    currlayer.clear();
    currlayer.reserve(topology[id] + 1);

    int totalOutputs = (id == topology.size() - 1) ? 0 : topology[id + 1];

    for (unsigned int i = 0; i < topology[id]; i++)
        currlayer.emplace_back(Neuron(i));
}
Network::Network(const vector<int> topology)
{
    int pipes[6][2];
    for(int i = 0;i < 6; i++)
        pipe(pipes[i]);

    id = 0;

    if (!fork())
    {
        close(pipes[4][0], pipes[5][1]);
        close(pipes[0][1], pipes[1][0], pipes[2][1], pipes[3][0]);
        LayersCreate(topology, pipes[0][0], pipes[1][1], pipes[2][0], 
                    pipes[3][1], pipes[5][0], pipes[4][1], id + 1 );
        close(pipes[0][0], pipes[1][1], pipes[2][0], pipes[3][1]);
        close(pipes[5][0], pipes[4][1]);
    }
    else
    {
        close(pipes[1][1], pipes[0][0], pipes[3][1], pipes[2][0]);
        close(pipes[5][0], pipes[4][1]);

        nextRead = pipes[1][0];
        lastRead = pipes[4][0];
        nextWrite = pipes[0][1];
        lastWrite = pipes[5][1];
        signalNextRead = pipes[3][0];
        signalNextWrite = pipes[2][1];
    }
    InitLayer(currlayer, topology, id);
    signal_lock(lastRead);

    printf("Network Creation Complete\n");
}
void Network::LayersCreate(const vector<int> &topology, const int pvRead, const int pvWrite,
    const int sigPR, const int sigPW, const int lw, const int lr, const int currId = 0)
{
    id = currId;
    prevRead = pvRead;
    prevWrite = pvWrite;
    signalPrevRead = sigPR;
    signalPrevWrite = sigPW;
    InitLayer(currlayer, topology, id);

    int pipes[4][2];
    for(int i = 0; i < 4; i++)
        pipe(pipes[i]);

    if (id < topology.size() - 1)
    {
        if (!fork())
        {
            close(signalPrevRead, signalPrevWrite);
            close(pipes[0][1], pipes[1][0], prevRead, prevWrite);
            LayersCreate(topology, pipes[0][0], pipes[1][1], pipes[2][0],
                         pipes[3][1], lw, lr, id + 1);
        }
        else
        {
            close(pipes[0][0], pipes[1][1], pipes[2][0], pipes[3][1]);
            close(lw, lr);
        }
    }
    else
    {
        lastRead = lw;
        lastWrite = lr;
        LastMain(topology);
        exit(0);
    }

    nextWrite = pipes[0][1];
    nextRead = pipes[1][0];
    signalNextRead = pipes[3][0];
    signalNextWrite = pipes[2][1];    
    LayerMain(topology);

    CONTEXTSWITCH;
    wait(NULL);
    exit(0);
}

void Network::LastMain(const vector<int> &topology)
{
    int size;
    char signal = 0;

    int firstRead = lastRead, firstWrite = lastWrite;
    signal_unlock(firstWrite);

    do {
        do {
            size = read(signalPrevRead, &signal, SIGNALSIZE);
        } while (size == 0);

        switch (signal) {
        
        case FRONTPROP:
            feedForwardHidden();
            cout << "Forward Propagation Result :" << currlayer[0].getOutputVal() << "\n";
            signal_unlock(firstWrite);
            CONTEXTSWITCH;
            break;

        case BACKPROP:
            double resVals[2];
            resVals[0] = backPropFunction1(currlayer[0].getOutputVal());
            resVals[1] = backPropFunction2(currlayer[0].getOutputVal());
            cout << getpid() <<  " " << id << " : " << resVals[0] << " " << resVals[1] << endl;
            write(prevWrite, &resVals[0], sizeof(double) * 2);
            write(signalPrevWrite, WBACKPROP, SIGNALSIZE);

            CONTEXTSWITCH;
            break;

        case DISPLAY:
            display(currlayer, id);
            signal_unlock(firstWrite);
            cout << "----------------------------\n";
            CONTEXTSWITCH;
            break;
        }
    } while (signal != EXIT);
    signal_unlock(firstWrite);
}
void Network::LayerMain(const vector<int> &topology)
{
    int size;
    char signal = 0;

    do {
        do {
            size = read(signalPrevRead, &signal, SIGNALSIZE);
            if (size == 0)
                size = read(signalNextRead, &signal, SIGNALSIZE);
            if (size == 0)
                CONTEXTSWITCH;
        } while (size == 0);

        switch (signal) {

        case FRONTPROP:
            feedForwardHidden();
            CONTEXTSWITCH;
            break;

        case BACKPROP:

            write(signalNextWrite, WBACKPROP, SIGNALSIZE);
            signal_lock(signalNextRead);
            feedBackwardHidden();
            CONTEXTSWITCH;
            break;

        case DISPLAY:
            display(currlayer, id);
            write(signalNextWrite, WDISPLAY, SIGNALSIZE);
            CONTEXTSWITCH;
            break;
        }
    } while (signal != EXIT);

    write(signalNextWrite, WEXIT, SIGNALSIZE);
}

void *initNeuron(void *args)
{
    NeuronArgs T = *(NeuronArgs *)args;
    Neuron& neuron = *T.neuron;
    neuron.setOutputVal(*T.inputVal);
    write(T.writeToNext, (char*)&neuron, sizeof(Neuron));
    pthread_exit(NULL);
}

void Network::feedForward(vector<double> &inputVals)
{
    assert(inputVals.size() == currlayer.size());

    pthread_t Threads[currlayer.size()];
    NeuronArgs args[currlayer.size()];

    for (int i = 0; i < currlayer.size(); i++) {
        args[i].copy(nextWrite, &inputVals[i], &currlayer[i]);
        pthread_create(&Threads[i], NULL, initNeuron, &args[i]);
    }
    for (int i = 0; i < currlayer.size(); i++)
        pthread_join(Threads[i], NULL);

    write(signalNextWrite, WFRONTPROP, SIGNALSIZE);

    signal_lock(lastRead);
    cout << "Forward Propagation Complete\n" << endl;
}

void *DotProduct(void *args)
{
    Thread T = *(Thread *)args;
    const Layer &layer = *T.threadlayer;
    const vector<vector<double>> &weights = *T.weights;
    Neuron& neuron = *T.neuron; 

    int i = neuron.getIndex(); 
    double sum = 0.0;
    for (unsigned n = 0; n < layer.size(); ++n)
        sum += layer[n].getOutputVal() * weights[n][i];

    neuron.setOutputVal(sum);
    write(T.writeToNext, (char*)&neuron, sizeof(Neuron));

    pthread_exit(0);
}
void Network::layerProcessing(const Layer &prevlayer, const int id)
{
    const vector<vector<double>>& edgeweights = weights[id];

    pthread_t Threads[currlayer.size()];
    Thread args[currlayer.size()];

    for (int i = 0; i < currlayer.size(); i++)
    {
        args[i].copy(nextWrite, &prevlayer, &edgeweights, &currlayer[i]);
        pthread_create(&Threads[i], NULL, DotProduct, &args[i]);
    }

    for (int i = 0; i < currlayer.size(); i++)
        pthread_join(Threads[i], NULL);
}
void Network::feedForwardHidden()
{
    Layer datalayer(topology[id - 1]);
    read(prevRead, &datalayer[0], sizeof(Neuron) * topology[id - 1]);

    Layer prevlayer;
    prevlayer.resize(topology[id - 1]);
    for(int i = 0; i < topology[id-1]; i++)
        prevlayer[datalayer[i].getIndex()] = 
            {datalayer[i].getIndex(), datalayer[i].getOutputVal()};

    layerProcessing(prevlayer, id - 1);
    display(currlayer, id);

    if (id != topology.size() - 1)
        write(signalNextWrite, WFRONTPROP, SIGNALSIZE);
}

void Network::feedBackward(vector<double>& resultVals)
{
    double resVals[2];
    write(signalNextWrite, WBACKPROP, SIGNALSIZE);
    signal_lock(signalNextRead);
    read(nextRead, &resVals, sizeof(double) * 2);

    resultVals[0] = resVals[0];
    resultVals[1] = resVals[1];
    cout << "Backward Propagation Results (" << resVals[0] << ", " << resVals[1] << ")\n";
    cout << "Backward Propagation Complete\n" << endl;
}
void Network::feedBackwardHidden()
{
    double resVals[2];
    read(nextRead, &resVals[0], sizeof(double) * 2);

    if(id == 0) return;
    cout << getpid() <<  " " << id << " : " << resVals[0] << " " << resVals[1] << endl;
    write(prevWrite, &resVals[0], sizeof(double) * 2);
    write(signalPrevWrite, WBACKPROP, SIGNALSIZE);
}
void Network::Print()
{
    display(currlayer, id);
    write(signalNextWrite, WDISPLAY, SIGNALSIZE);
    signal_lock(lastRead);
}
void Network::End()
{
    char temp;
    write(signalNextWrite, WEXIT, SIGNALSIZE);
}

int main()
{
    // weights = {
    //     {
    //         {0.1, -0.2, 0.3, 0.1, -0.2, 0.3, 0.1, -0.2},
    //         {-0.4, 0.5, 0.6, -0.4, 0.5, 0.6, -0.4, 0.5}
    //     },
    //     {
    //         {-0.2, 0.3, -0.4, 0.5, 0.6, -0.7, 0.8, -0.9}, 
    //         {0.1, 0.9, -0.3, 0.2, -0.5, 0.4, 0.6, -0.8},
    //         {-0.7, 0.5, 0.8, -0.2, -0.3, -0.6, 0.1, 0.4},
    //         {-0.2, 0.3, -0.4, 0.5, 0.6, -0.7, 0.8, -0.9},
    //         {0.1, 0.9, -0.3, 0.2, -0.5, 0.4, 0.6, -0.8},
    //         {-0.7, 0.5, 0.8, -0.2, -0.3, -0.6, 0.1, 0.4},
    //         {-0.2, 0.3, -0.4, 0.5, 0.6, -0.7, 0.8, -0.9},
    //         {0.1, 0.9, -0.3, 0.2, -0.5, 0.4, 0.6, -0.8}
    //     },
    //     {
    //         {0.2, -0.3, 0.4, -0.5, -0.6, 0.7, -0.8, 0.9},
    //         {-0.1, -0.9, 0.3, -0.2, 0.5, -0.4, -0.6, 0.8},
    //         {0.7, -0.5, -0.8, 0.2, 0.3, 0.6, -0.1, -0.4},
    //         {0.2, -0.3, 0.4, -0.5, -0.6, 0.7, -0.8, 0.9},
    //         {-0.1, -0.9, 0.3, -0.2, 0.5, -0.4, -0.6, 0.8},
    //         {0.7, -0.5, -0.8, 0.2, 0.3, 0.6, -0.1, -0.4},
    //         {0.2, -0.3, 0.4, -0.5, -0.6, 0.7, -0.8, 0.9},
    //         {-0.1, -0.9, 0.3, -0.2, 0.5, -0.4, -0.6, 0.8}
    //     },
    //     {
    //         {0.3, -0.4, 0.5, -0.6, -0.7, 0.8, -0.9, 0.1},
    //         {-0.2, -0.9, 0.4, -0.3, 0.5, -0.6, -0.8, 0.1},
    //         {0.6, -0.5, -0.7, 0.2, 0.4, 0.8, -0.1, -0.3},
    //         {0.3, -0.4, 0.5, -0.6, -0.7, 0.8, -0.9, 0.1},
    //         {-0.2, -0.9, 0.4, -0.3, 0.5, -0.6, -0.8, 0.1},
    //         {0.6, -0.5, -0.7, 0.2, 0.4, 0.8, -0.1, -0.3},
    //         {0.3, -0.4, 0.5, -0.6, -0.7, 0.8, -0.9, 0.1},
    //         {-0.2, -0.9, 0.4, -0.3, 0.5, -0.6, -0.8, 0.1}
    //     },
    //     {
    //         {0.4, -0.5, 0.6, -0.7, -0.8, 0.9, -0.1, 0.2},
    //         {-0.3, -0.8, 0.5, -0.4, 0.6, -0.7, -0.9, 0.2},
    //         {0.5, -0.4, -0.6, 0.3, 0.2, 0.8, -0.2, -0.1},
    //         {0.4, -0.5, 0.6, -0.7, -0.8, 0.9, -0.1, 0.2},
    //         {-0.3, -0.8, 0.5, -0.4, 0.6, -0.7, -0.9, 0.2},
    //         {0.5, -0.4, -0.6, 0.3, 0.2, 0.8, -0.2, -0.1},
    //         {0.4, -0.5, 0.6, -0.7, -0.8, 0.9, -0.1, 0.2},
    //         {-0.3, -0.8, 0.5, -0.4, 0.6, -0.7, -0.9, 0.2}
    //     },
    //     {
    //         {0.5, -0.6, 0.7, -0.8, -0.9, 0.1, -0.2, 0.3},
    //         {-0.4, -0.7, 0.6, -0.5, 0.8, -0.6, -0.2, 0.1},
    //         {0.4, -0.3, -0.5, 0.1, 0.6, 0.7, -0.3, -0.2},
    //         {0.5, -0.6, 0.7, -0.8, -0.9, 0.1, -0.2, 0.3},
    //         {-0.4, -0.7, 0.6, -0.5, 0.8, -0.6, -0.2, 0.1},
    //         {0.4, -0.3, -0.5, 0.1, 0.6, 0.7, -0.3, -0.2},
    //         {0.5, -0.6, 0.7, -0.8, -0.9, 0.1, -0.2, 0.3},
    //         {-0.4, -0.7, 0.6, -0.5, 0.8, -0.6, -0.2, 0.1}
    //     },
    //     {
    //         {-0.1},
    //         {0.2},
    //         {0.3},
    //         {0.4},
    //         {0.5},
    //         {-0.6},
    //         {-0.7},
    //         {0.8}
    //     }
    // }; 
    // topology = {2, 8, 8, 8, 8, 8, 8, 1};

    //vector<double> inputs = {0.1, 0.2};
    ReadFile();

    Network net(topology);
    net.Print();

    for(int i = 0; i < ITERATIONS; i++) {
        net.feedForward(inputs);
        net.feedBackward(inputs);
    }

    net.End();
    wait(NULL);
    return 0;
}

/*

* *
0{2, 8}
* * * * * * * * 
1{8, 8}
* * * * * * * *
2{8, 8}
* * * * * * * *
3{8, 8}
* * * * * * * *
4{8, 8}
* * * * * * * *
5{8, 8}
* * * * * * * *
1{8, 1}
*

*/
