#include <mpi.h>
#include <pthread.h>
#include <cstdlib>
#include <cmath>
#include <vector>

constexpr int SUCCESS = 0;
constexpr int FAIL = 1;
constexpr int REQUEST_TAG = 2;
constexpr int ANSWER_TAG = 3;
constexpr int NEED_TASKS = 4;
constexpr int TURN_OFF = 5;

constexpr int TASKS_IN_LIST = 470;
constexpr int L = 1500;
constexpr int ITERATION = 5;

typedef struct Task {
  long repeatNum;
} Task;

struct ProcessSharedData {
  int procRank = 0;
  int commSize = 0;
  int iterCounter = 0;
  int curTaskNum = 0;
  int taskListSize = 0;
  double globalRes = 0;
  double summaryDisbalance = 0;
  std::vector<Task> taskList;
  pthread_mutex_t mutex = {0};
};

void generateTaskList(ProcessSharedData *psd) {
  psd->taskListSize = TASKS_IN_LIST;
  for (int i = psd->procRank * TASKS_IN_LIST; i < (psd->procRank + 1) * TASKS_IN_LIST; i++) {
    long repeatNum = std::abs(TASKS_IN_LIST / 2 - i % TASKS_IN_LIST)
                     * std::abs(psd->procRank + 1 - (psd->iterCounter % psd->commSize)) * L;
    psd->taskList[i % TASKS_IN_LIST].repeatNum = std::abs(repeatNum);

  }
}

int getTaskFrom(int from, ProcessSharedData *psd) {
  int flag = NEED_TASKS;
  MPI_Send(&flag, 1, MPI_INT, from, REQUEST_TAG, MPI_COMM_WORLD);
  MPI_Recv(&flag, 1, MPI_INT, from, ANSWER_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  if (flag == FAIL) {
    return FAIL;
  }
  Task receiveTask;
  MPI_Recv(&receiveTask, 1, MPI_INT, from, ANSWER_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  pthread_mutex_lock(&psd->mutex);
  psd->taskList[psd->taskListSize] = receiveTask;
  psd->taskListSize++;
  pthread_mutex_unlock(&psd->mutex);
  return SUCCESS;
}

void doTask(Task task, ProcessSharedData *psd) {
  for (long i = 0; i < task.repeatNum; i++) {
    psd->globalRes += sin(i);
  }
}

void *routineSenderThread(void *processSharedDataArg) {
  auto *psd = static_cast<ProcessSharedData *> (processSharedDataArg);
  int flag;
  while (psd->iterCounter < ITERATION) {
    MPI_Status status;
    MPI_Recv(&flag, 1, MPI_INT, MPI_ANY_SOURCE, REQUEST_TAG, MPI_COMM_WORLD, &status);
    if (flag == TURN_OFF) break;
    pthread_mutex_lock(&psd->mutex);
    if (psd->curTaskNum >= psd->taskListSize - 1) {
      pthread_mutex_unlock(&psd->mutex);
      flag = FAIL;
      MPI_Send(&flag, 1, MPI_INT, status.MPI_SOURCE, ANSWER_TAG, MPI_COMM_WORLD);
      continue;
    }
    psd->taskListSize--;
    Task sendTask = psd->taskList[psd->taskListSize];
    pthread_mutex_unlock(&psd->mutex);
    flag = SUCCESS;
    MPI_Send(&flag, 1, MPI_INT, status.MPI_SOURCE, ANSWER_TAG, MPI_COMM_WORLD);
    MPI_Send(&sendTask, 1, MPI_INT, status.MPI_SOURCE, ANSWER_TAG, MPI_COMM_WORLD);
  }
  return nullptr;
}

void *routineExecutorThread(void *processSharedDataArg) {
  MPI_Barrier(MPI_COMM_WORLD);
  auto *psd = static_cast<ProcessSharedData *> (processSharedDataArg);
  double start, end;
  psd->iterCounter = 0;
  while (psd->iterCounter < ITERATION) {
    int countOfFinishedTasks = 0;
    int hasTasks = 1;
    pthread_mutex_lock(&psd->mutex);
    psd->curTaskNum = 0;
    generateTaskList(psd);
    pthread_mutex_unlock(&psd->mutex);
    start = MPI_Wtime();
    while (hasTasks) {
      pthread_mutex_lock(&psd->mutex);
      if (psd->curTaskNum < psd->taskListSize) {
        Task task = psd->taskList[psd->curTaskNum];
        pthread_mutex_unlock(&psd->mutex);
        doTask(task, psd);
        countOfFinishedTasks++;
        pthread_mutex_lock(&psd->mutex);
        psd->curTaskNum++;
        pthread_mutex_unlock(&psd->mutex);
        continue;
      }
      psd->curTaskNum = 0;
      psd->taskListSize = 0;
      pthread_mutex_unlock(&psd->mutex);
      hasTasks = 0;
      for (int i = 0; i < psd->commSize; i++) {
        if (i == psd->procRank) continue;
        if (getTaskFrom(i, psd) == SUCCESS) {
          hasTasks = 1;
        }
      }
    }
    end = MPI_Wtime();
    double timeTaken = end - start;
    double minTime, maxTime;
    MPI_Reduce(&timeTaken, &minTime, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&timeTaken, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    pthread_mutex_lock(&psd->mutex);
    psd->summaryDisbalance += (maxTime - minTime) / maxTime;
    pthread_mutex_unlock(&psd->mutex);
    MPI_Barrier(MPI_COMM_WORLD);
    psd->iterCounter++;
  }
  int flag = TURN_OFF;
  MPI_Send(&flag, 1, MPI_INT, psd->procRank, REQUEST_TAG, MPI_COMM_WORLD);
  return nullptr;
}

int main(int argc, char **argv) {
  int isMultithreadingProvided;
  MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &isMultithreadingProvided);
  if (isMultithreadingProvided != MPI_THREAD_MULTIPLE) {
    MPI_Finalize();
    std::cerr << "Unable to init MPI with MPI_THREAD_MULTIPLE level support" << std::endl;
    return 0;
  }

  auto *psd = new ProcessSharedData();
  MPI_Comm_rank(MPI_COMM_WORLD, &psd->procRank);
  MPI_Comm_size(MPI_COMM_WORLD, &psd->commSize);
  psd->taskList = std::vector<Task>(psd->commSize * TASKS_IN_LIST);
  pthread_t threads_id[2];
  pthread_mutex_init(&psd->mutex, nullptr);
  pthread_attr_t attrs;

  if (pthread_attr_init(&attrs) != 0) {
    delete psd;
    MPI_Finalize();
    std::cerr << "Unable to initialize attributes in process: " << psd->procRank << std::endl;
    return 0;
  }

  if (pthread_attr_setdetachstate(&attrs, PTHREAD_CREATE_JOINABLE) != 0) {
    delete psd;
    MPI_Finalize();
    std::cerr << "Error in setting attributes in process: " << psd->procRank << std::endl;
    return 0;
  }

  if (pthread_create(&threads_id[0], &attrs, routineSenderThread, psd) != 0 ||
      pthread_create(&threads_id[1], &attrs, routineExecutorThread, psd) != 0) {
    delete psd;
    MPI_Finalize();
    std::cerr << "Unable to create a thread in process: " << psd->procRank << std::endl;
    return 0;
  }
  double start, end;
  if (psd->procRank == 0) {
    start = MPI_Wtime();
  }


  for (auto thread: threads_id) {
    if (pthread_join(thread, nullptr) != 0) {
      delete psd;
      MPI_Finalize();
      std::cerr << "Unable to join a thread in process: " << psd->procRank << std::endl;
      return 0;
    }
  }

  if (psd->procRank == 0) {
    end = MPI_Wtime();
    double timeTaken = end - start;
    std::cout << "Summary disbalance: " << psd->summaryDisbalance / (ITERATION) * 100 << std::endl;
    std::cout << "Global time: " << timeTaken << std::endl;
  }

  pthread_attr_destroy(&attrs);
  pthread_mutex_destroy(&psd->mutex);
  delete psd;
  MPI_Finalize();
  return 0;
}