# Л.р. №1 (Сахаров А., ФИИТ КГ)

Руководство по запуску:
1. Клонировать, собрать cmake-ом или создать проект в студии на основе CMakeLists.
2. Добавить пути к библиотекам GTEST, PTHREADS, если они отсутствуют в путях ОС по умолчанию (у меня они раскиданы в /usr/lib).
Если не планируется запускать тесты, то второй проект можно выпилить из смака, дабы не мешался. Основная программа никаких потусторонних зависимостей не тянет.
3. Запустить программу можно с ключом --help или -h, чтобы вывелось описание необходимых аргументов командной строки.

Usage: <exec_name> H L Err Epch TD TL VD VL S

  Replace each var with corresponding argument, where
  
  
  	H - number of neurons in hidden layer
  	L - learning rate for gradient descend opt method
  	Err - desired cross-entropy accuracy
  	Epch - number of epochs
  	TD - path to file with train data
  	TL - path to file with train labels
  	VD - path to file with validation data
  	VL - path to file with validation labels
  	S - path to file in which network params will be stored
    
  
  For H L Err and Epch zero value can be used. In this case default values are as following:
  
  
  	200 neurons in hidden layer
  	0.01 as learning rate for gradient descend opt method
  	0.005 as desired cross-entropy accuracy
  	10 train epochs
    
    
  If S is not specified, default name "vanilla.params" will be used
