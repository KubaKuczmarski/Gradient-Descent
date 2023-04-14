**[PL]**

# ALGORYTM GRADINETU PROSTEGO

Algorytm gradientu prostego to jedna z najczęściej stosowanych metod optymalizacji w uczeniu maszynowym. Polega on na minimalizacji funkcji celu poprzez krokowe przesuwanie się w kierunku przeciwnym do kierunku gradientu tej funkcji w danym punkcie.

Kroki algorytmu gradientu prostego:
  1. Inicjalizacja punktu startowego.
  2. Obliczenie gradientu funkcji w punkcie startowym.
  3. Określenie kierunku przeciwnego do gradientu.
  4. Wykonanie kroku w kierunku przeciwnym do gradientu, zgodnie z wybranym rozmiarem kroku (tzw. learning rate)
  5. Powtarzanie kroków 2-4 aż do osiągnięcia zadowalającego minimum funkcji.

Założenia:
  - Funkcja jest ciągła i różniczkowalna
  - Funkcja jest wypukła w badanej dziedzinie


# ZADANIE

Tematem projektu jest implementacja algorytmu gradientu prostego oraz zastosowanie go do znalezienia minimum funkcji f(x) i g(x) oraz zbadanie wpływu
rozmiaru kroku dla różnych (losowych) punktów początkowych.

# BADANE FUNKCJE

Funkcja f(x)

![image](https://user-images.githubusercontent.com/113121214/232006758-48c4ae53-ebfd-4ca3-84f4-33632a182647.png)

Funkcja g(x)

![image](https://user-images.githubusercontent.com/113121214/232006663-4e58131f-4578-4e82-8b6e-f40ef7954b30.png)



=========================================================================================

**[ENG]**

# GRADIENT DESCENT ALGORITHM 

The simple gradient algorithm is one of the most commonly used optimization methods in machine learning. It involves minimizing the objective function by gradually moving in the opposite direction of the gradient of that function at a given point.

Steps of the simple gradient algorithm:
  1. Initialization of the starting point.
  2. Calculation of the gradient of the function at the starting point.
  3. Determination of the opposite direction to the gradient.
  4. Taking a step in the opposite direction of the gradient, according to the chosen step size (learning rate).
  5. Repeating steps 2-4 until a satisfactory minimum of the function is achieved.

Assumptions:
  - The function is continuous and differentiable.
  - The function is convex in the investigated domain.

# TASK
The project's topic is to implement the simple gradient algorithm and apply it to finding the minimum of functions f and g, and to investigate the influence of the step size for different (random) starting points.

# INVESTIGATED FUNCTIONS

Function f(x)

![image](https://user-images.githubusercontent.com/113121214/232006758-48c4ae53-ebfd-4ca3-84f4-33632a182647.png)

Function g(x)

![image](https://user-images.githubusercontent.com/113121214/232006663-4e58131f-4578-4e82-8b6e-f40ef7954b30.png)
