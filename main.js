const KNN = require('ml-knn')
const csv = require('csvtojson')
const prompt = require('prompt')
const knn = new KNN()

const csvFilePath = 'iris.csv'
const names = ['sepalLength', 'sepalWidth', 'petalLength', 'petalWidth', 'type']
let separationSize
let learningRate = 0.7
let data = [],
    X = [],
    y = []

let trainingSetX = [],
    trainingSetY = [],
    testSetX = [],
    testSetY = []

csv({ noheader: true, headers: names }).fromFile(csvFilePath).on('json', (jsonObj) => {
    data.push(jsonObj)
}).on('done', (err) => {
    console.log("Done")
    if (err) {
        throw err
    }

    separationSize = learningRate * data.length
    data = shuffleArray(data)
    dressData()
})

function dressData() {
    console.log("Dressing data...")
        /* Data will be organised as follow: 
        - 0: setosa
        - 1: versicolor
        - 2: virginica
         */

    let types = new Set() // For unique classes
    data.forEach((row) => {
        types.add(row.type)
    })

    typesArray = [...types] //Save different types of classes
    data.forEach((row) => {
        let rowArray, typeNumber
        rowArray = Object.keys(row).map(key => parseFloat(row[key])).slice(0, 4)
        typeNumber = typesArray.indexOf(row.type)

        X.push(rowArray)
        y.push(typeNumber)
    })

    trainingSetX = X.slice(0, separationSize)
    trainingSetY = y.slice(0, separationSize)
    testSetX = X.slice(separationSize)
    testSetY = y.slice(separationSize)

    train()
}

function shuffleArray(array) {
    for (var i = array.length - 1; i > 0; i--) {
        var j = Math.floor(Math.random() * (i + 1));
        var temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }

    return array;
}

function train() {
    knn.train(trainingSetX, trainingSetY)
    test()
}

function test() {
    const result = knn.predict(testSetX)
    const testSetLength = testSetX.length
    const predictError = error(result, testSetY)
    console.log(`Test Set Size = ${testSetLength} and number of Misclassifications = ${predictError}`)
    predict()
}

function error(predicted, expected) {
    let misclassifications = 0
    for (var i = 0; i < predicted.length; i++) {
        if (predicted[i] !== expected[i]) {
            misclassifications++
        }
    }

    return misclassifications
}

function predict() {
    let temp = []
    prompt.start()
    prompt.get(['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'], (err, result) => {
        if (!err) {
            for (var key in result) {
                temp.push(parseFloat(result[key]))
            }

            console.log(`With ${temp} -- type=${knn.getSinglePrediction(temp)}`)
        }
    })
}