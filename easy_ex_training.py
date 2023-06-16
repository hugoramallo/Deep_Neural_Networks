#This is an easy example of a basic training with mini batch gradient or estochastic

for epoch in range(100): #100 entrenamientos
    for x,y in trainloader: #obtenemos las muestras para cada lote
        yhat = model(x) #obtenemos la predicción
        loss = criterion(yhat,y ) #calculamos la pérdida
        optimizer.zero-grad() #establecemos gradiente a 0
        loss.backward() #diferenciamos la pérdida con respecto a los parámetros
        optimizer.step() #actualizamos parámetros
        #puntos optimizador
        w.data = w.data-lr * w.grad.data
        b.data = w.data - lr * b.grad.data
