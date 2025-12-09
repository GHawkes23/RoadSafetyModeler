package cs455.rsm;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.regression.RandomForestRegressionModel;
import org.apache.spark.ml.regression.RandomForestRegressor;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import static org.apache.spark.sql.functions.col;

public class Modeling {

    public static Dataset<Row> randomForestTrain(Dataset<Row> allFeatures) {

        // Basic cleaning: keep only rows that actually have a crash_count label
        Dataset<Row> data = allFeatures.filter(col("crash_count").isNotNull());

        // Build feature vector: all numeric columns except grid_id + crash_count
        String[] allCols = data.columns();

        List<String> featureColsList = Arrays.stream(allCols)
                .filter(c -> !c.equals("grid_id"))
                .filter(c -> !c.equals("crash_count"))
                .collect(Collectors.toList());

        String[] featureCols = featureColsList.toArray(new String[0]);

        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(featureCols)
                .setOutputCol("features");

        Dataset<Row> assembled = assembler.transform(data)
                .select("grid_id", "crash_count", "features");

        // Train/test split
        Dataset<Row>[] splits = assembled.randomSplit(new double[]{0.8, 0.2}, 42);
        Dataset<Row> train = splits[0];
        Dataset<Row> test  = splits[1];

        // Random forest regressor
        RandomForestRegressor rf = new RandomForestRegressor()
                .setLabelCol("crash_count")
                .setFeaturesCol("features")
                .setNumTrees(100);

        RandomForestRegressionModel model = rf.fit(train);

        // Evaluate on test set
        Dataset<Row> testPredictions = model.transform(test);

        RegressionEvaluator evaluator = new RegressionEvaluator()
                .setLabelCol("crash_count")
                .setPredictionCol("prediction")
                .setMetricName("rmse");

        double rmse = evaluator.evaluate(testPredictions);
        System.out.println("RandomForest RMSE on held-out test: " + rmse);

        testPredictions.select("grid_id", "crash_count", "prediction").show(20, false);

        // Use the same model to predict crash_count for ALL feature rows
        Dataset<Row> fullPredictions = model.transform(assembled);

        // This has: grid_id, crash_count, features, prediction
        return fullPredictions;
    }
}
