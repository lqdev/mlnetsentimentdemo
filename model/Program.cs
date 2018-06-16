using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Models;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;

namespace model
{
    class Program
    {
        static void Main(string[] args)
        {
            string dataPath = "./training-data.tsv";
            string modelPath = "./model.zip";
            PredictionModel<SentimentData, SentimentPrediction> model = Train(dataPath, modelPath);

            IEnumerable<SentimentData> sentiments = new[]
            {
                new SentimentData
                {
                    SentimentText = "Please refrain from adding nonsense to Wikipedia."
                },
                new SentimentData
                {
                    SentimentText = "He is the best, and the article should say that."
                }
            };

            var predictions = model.Predict(sentiments);

            //Workaround for C# version under 7.1
            for (int i = 0; i < sentiments.Count(); i++)
            {
                Console.WriteLine($"Sentiment: {sentiments.ElementAt(i).SentimentText} - Prediction: {(predictions.ElementAt(i).Sentiment ? "Positive" : "Negative")}");
            }

        }

        static PredictionModel<SentimentData, SentimentPrediction> Train(string dataPath, string modelPath)
        {
            LearningPipeline pipeline = new LearningPipeline();

            pipeline.Add(new TextLoader(dataPath).CreateFrom<SentimentData>());

            pipeline.Add(new TextFeaturizer("Features", "SentimentText"));

            pipeline.Add(new FastForestBinaryClassifier() { NumLeaves = 5, NumTrees = 5, MinDocumentsInLeafs = 2 });

            PredictionModel<SentimentData, SentimentPrediction> model = pipeline.Train<SentimentData, SentimentPrediction>();

            model.WriteAsync(modelPath);

            return model;

        }
    }
}
