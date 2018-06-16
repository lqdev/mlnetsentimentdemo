using Microsoft.ML.Runtime.Api;

namespace model
{
    public class SentimentPrediction
    {
        [ColumnName("PredictedLabel")]
        public bool Sentiment;
    }
}