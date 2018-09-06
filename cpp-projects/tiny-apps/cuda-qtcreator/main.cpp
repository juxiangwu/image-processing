#include <QCoreApplication>
extern "C"
int cuda_main();
int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);
    cuda_main();
    return a.exec();
}
