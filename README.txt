Pueden correr los archivos measure_own_ADC.py, measure_QR.py, measure_reference_ADC.py, para medir los métodos de el ADC implementado por nosotros, el QR de LAPACK y un método muy similar a ADC en LAPACK, respectivamente. Les dará los errores medidos y normalizados al igual que especifica el informe, como también los tiempos de ejecución que tarda cada uno.

En la carpeta deprecated/ y 100digits/ están los métodos alternativos a FMM que intentamos usar pero fallaron catastróficamente con la precisión. (la carpeta 100digits simula los métodos de deprecated pero con un tipo de 100 dígitos de precisión para verificar correctitud con N grande, así que los métodos principales están en deprecated/). De igual manera estas carpetas no son usadas en el algoritmo final.

Como pueden ver desde los archivos measure_X.pi, estos llaman a test_matrices.py para obtener las matrices de prueba con sus nombres, luego de test_script.py obtiene las funciones para medir un método, dada la función solver y la función de normalización del error.

measure_own_ADC.py llama a proyecto.py, que sería el archivo principal que desencadena en los otros archivos para resolver cada parte del algoritmo, como describen los nombres de los archivos/funciones.

Si quieren ver la función principal que devuelve la diagonalización, esta está justamente en proyecto.py y se llama DnC_SimTridiag, pero igualmente recomendamos solo usar los archivos de measure_own_ADC.py etc.

La función DnC_SimTridiag recibe una matriz cuadrada simétrica tridiagonal A, y opcionalmente, la tolerancia utilizada para la deflación (más específicamente, el eta múltiplo de épsilon). Devuelve el par D, Q, que cumplen A = Q D Q^T, con Q unitaria y D diagonal.