# Л.р. №1 (Сахаров А., ФИИТ КГ)

Руководство по запуску:
1. Клонировать, собрать cmake-ом или создать проект в студии на основе CMakeLists.
2. Запустить программу можно с ключом --help или -h, чтобы вывелось описание необходимых аргументов командной строки.

Использование: <exec_name> H L Err Epch TD TL VD VL S
Замените каждую из переменных на соответствующее значение, согласно описаниям ниже:
  
  
  	H - число нейронов в скрытом слое
  	L - learning rate
  	Err - остановочная точность кросс-энтропии
  	Epch - число эпох
  	TD - путь к файлу с тренировочными данными
  	TL - путь к файлу с метками для тренировочных данных
  	VD - путь к файлу с тестовыми данными
  	VL - путь к файлу с метками для тестовых данных
  	S - путь и имя файла, в который будут сохранены параметры сети (НЕ ИСПОЛЬЗУЕТСЯ)
    
  
  Для H L Err и Epch можно использовать нулевые значения. Тогда используются значения по умолчанию:
  
  
  	200 нейронов в скрытом слое
  	0.01 learning rate
  	0.005 остановочная точность кросс-энтропии
  	10 тренировочных эпох
        
  Пример: run.exe 300 0.05 0 0 ../data/train_images ../data/train_labels ../data/test_images ../data/test_labels
  
  Отчёт приложен в папке с кодом.
