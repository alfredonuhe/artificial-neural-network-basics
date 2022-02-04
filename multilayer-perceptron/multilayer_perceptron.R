library(RSNNS)
set.seed(1)

# Carga de los datos
# Load data 

fold <- 3
trainSet <- read.csv(paste("train",fold,".txt",sep=""),dec=".",sep=" ",header = F)
testSet  <- read.csv(paste("test", fold,".txt",sep=""),dec=".",sep=" ",header = F)

# Seleccion de la salida
# Select output column#SELECT OUTPUT COLUMN
nTarget <- ncol(trainSet)

# Separar entrada de la salida
# Separate input and output
trainInput <- trainSet[,-nTarget]
testInput <-  testSet[,-nTarget]

# Transformar la salida discreta a numerica
# Transform discrete output to numeric
trainTarget <- decodeClassLabels(trainSet[,nTarget])
testTarget <-  decodeClassLabels(testSet[,nTarget])

# Transformar las entradas de dataframe a matrix para mlp
# Transform inputs to matrix
trainInput <- as.matrix(trainInput)
testInput  <- as.matrix(testInput)

# Seleccion de los parametros
# Select parameters
topologia        <- c(5,5)
razonAprendizaje <- 0.05
ciclosMaximos    <- 5000

# Asignar nombre de fichero
# Name file
fileID <- paste("fX",fold,"_topX",paste(topologia,collapse="-"),"_ra",razonAprendizaje,"_CMX",ciclosMaximos,".csv",sep="")

# Aprendizaje
# Train
model <- mlp(x= trainInput,
             y= trainTarget,
             inputsTest= testInput,
             targetsTest= testTarget,
             size= topologia,
             maxit=ciclosMaximos,
             learnFuncParams=c(razonAprendizaje),
             shufflePatterns = F
)

# Grafico de la evolucion del error
# Graph error evolution
plotIterativeError(model)

# Generar las predicciones en bruto (valores reales)
# Generate predictions
trainPred <- predict(model,trainInput)
testPred  <- predict(model,testInput)

# Calculo de las matrices de confusion
# Calculate confusion matrix
trainCm <- confusionMatrix(trainTarget,trainPred)
testCm  <- confusionMatrix(testTarget,testPred)

trainCm
testCm

# Porcentaje total de aciertos
# Total accuracy rate
accuracy <- function (cm) sum(diag(cm))/sum(cm)
accuracies <- c(TrainAccuracy= accuracy(trainCm), TestAccuracy=  accuracy(testCm) )
print(accuracies)

# Tabla con los errores por ciclo
# Errors per cicle table
iterativeErrors <- data.frame(MSETrain= (model$IterativeFitError/nrow(trainSet)),
                     MSETest= (model$IterativeTestError/nrow(testSet)))

# Calcular errores finales MSE
# Calculate MSE
MSEtrain <-sum((trainTarget - trainPred)^2)/nrow(trainSet)
MSEtest <-sum((testTarget - testPred)^2)/nrow(testSet)

# Calcular la clase de salida
# Calculate output class
trainPredClass<-as.factor(apply(trainPred,1,which.max))  
testPredClass<-as.factor(apply(testPred,1,which.max)) 

# Guardado de resultados
# Save results
saveRDS(model,             paste("nnet_",gsub("\\.csv","",fileID),".rds",sep=""))
write.csv(accuracies,     paste("finalAccuracies_",fileID,sep=""))
write.csv(iterativeErrors,paste("iterativeErrors_",fileID,sep=""))
# Salidas de test en bruto
# Test outputs
write.csv(testPred ,       paste("TestRawOutputs_",fileID,sep=""), row.names = FALSE)
write.csv(testPredClass,   paste("TestClassOutputs_",fileID,sep=""),row.names = FALSE)
# Matrices de confusi?n
# Confusion matrix
write.csv(trainCm,        paste("trainCm_",fileID,sep=""))
write.csv(testCm,         paste("testCm_",fileID,sep="")) 