package cs455.rsm;

import java.util.Arrays;

import org.apache.sedona.sql.utils.SedonaSQLRegistrator;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import static org.apache.spark.sql.functions.col;
import static org.apache.spark.sql.functions.count;
import static org.apache.spark.sql.functions.expr;
import static org.apache.spark.sql.functions.sum;

public class Main {

    private static final String INPUT_BASE = "data/";
    private static final String LARIMER_GRID_FILE = INPUT_BASE + "LarimerGrids.parquet";
    private static final String ROADS_FILE = INPUT_BASE + "RoadCenterLines.parquet";
    private static final String CRASHES_FILE = INPUT_BASE + "Crashes.parquet";
    private static final String CURBSGUTTERS_FILE = INPUT_BASE + "CurbsAndGutters.parquet";
    private static final String GUARDRAILS_FILE = INPUT_BASE + "Guardrails.parquet";
    private static final String MEDIANS_FILE = INPUT_BASE + "Medians.parquet";
    private static final String SIDEWALKS_FILE = INPUT_BASE + "Sidewalks.parquet";
    private static final String SIGNPANELS_FILE = INPUT_BASE + "SignPanels.parquet";
    private static final String SIGNMOUNTS_FILE = INPUT_BASE + "SignMounts.parquet";

    public static void main(String[] args) {

        // Create SparkSession and register Sedona
        SparkSession spark = SparkSession.builder()
                .appName("LarimerRoadSafety")
                .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
                .config("spark.kryo.registrator",
                        "org.apache.sedona.core.serde.SedonaKryoRegistrator")
                .getOrCreate();

        spark.sparkContext().setLogLevel("WARN");
        SedonaSQLRegistrator.registerAll(spark);

        System.out.println("\n=== Spark + Sedona initialized ===\n");

        // Load all layers from Parquet and add Sedona geometry column `geom`
        Dataset<Row> larimerGrid = loadParquetWithGeom(spark, LARIMER_GRID_FILE);
        Dataset<Row> roads = loadParquetWithGeom(spark, ROADS_FILE);
        Dataset<Row> crashes = loadParquetWithGeom(spark, CRASHES_FILE);
        Dataset<Row> curbsGutters = loadParquetWithGeom(spark, CURBSGUTTERS_FILE);
        Dataset<Row> guardrails = loadParquetWithGeom(spark, GUARDRAILS_FILE);
        Dataset<Row> medians = loadParquetWithGeom(spark, MEDIANS_FILE);
        Dataset<Row> sidewalks = loadParquetWithGeom(spark, SIDEWALKS_FILE);
        Dataset<Row> signPanels = loadParquetWithGeom(spark, SIGNPANELS_FILE);
        Dataset<Row> signMounts = loadParquetWithGeom(spark, SIGNMOUNTS_FILE);

        printSchemaAndCount("Larimer grid", larimerGrid);
        printSchemaAndCount("Roads", roads);
        printSchemaAndCount("Crashes", crashes);
        printSchemaAndCount("Curb & Gutter", curbsGutters);
        printSchemaAndCount("Guardrails", guardrails);
        printSchemaAndCount("Medians", medians);
        printSchemaAndCount("Sidewalks", sidewalks);
        printSchemaAndCount("Sign Panels", signPanels);
        printSchemaAndCount("Sign Mounts", signMounts);

        Dataset<Row> roadFeatures = FeatureEngineering.aggregateRoadLengthsBySpeed(spark, larimerGrid, roads, "id");
        // //System.out.println("\n=== Road lenghts by speed limit features (per grid) ===");
        // //roadFeatures.show(10, false);
        // Dataset<Row> signPanelFeatures = FeatureEngineering.aggregateSignsFullPivot(spark, larimerGrid, signPanels, "id", "MUTCDDESCD");
        // //System.out.println("\n=== Sign panel features (per grid) ===");
        // //signPanelFeatures.show(10, false);
        Dataset<Row> signPanelsSlim = signPanels.select("OBJECTID", "MUTCDDESCD", "geom");
        Dataset<Row> signMountsSlim = signMounts.select("OBJECTID", "MUTCDDESCD", "geom");
        Dataset<Row> allSigns = signPanelsSlim.union(signMountsSlim);

        // Aggregate sign features once over the combined signs, apparently the boxes of sand we've made to do calculations can't distinguish "lane_row_control" from "lane_row_control" (They're the same thing)
        // Because both sign classes have the same types, the count should be combined, but I'm leaving that commented block above in case we need it again.
        Dataset<Row> signFeatures = FeatureEngineering.aggregateSignsFullPivot(spark, larimerGrid, allSigns, "id", "MUTCDDESCD");

        System.out.println("\n=== Combined sign features (panels + mounts) per grid ===");
        signFeatures.show(10, false);
        // Dataset<Row> signMountFeatures = FeatureEngineering.aggregateSignsFullPivot(spark, larimerGrid, signMounts, "id", "MUTCDDESCD");
        //System.out.println("\n=== Sign panel and Sign mount features (per grid) ===");
        //signMountFeatures.show(10, false);
        Dataset<Row> guardrailFeatures = FeatureEngineering.aggregateGuardrailLengthsByMaterial(spark, larimerGrid, guardrails, "id");
        //System.out.println("\n=== Guardrail length by material (per grid) ===");
        //guardrailFeatures.show(10, false);
        Dataset<Row> medianFeatures = FeatureEngineering.aggregateMedianLengthsByType(spark, larimerGrid, medians, "id");
        Dataset<Row> sidewalkFeatures = FeatureEngineering.aggregateSidewalkLengths(spark, larimerGrid, sidewalks, "id");
        Dataset<Row> crashFeatures = FeatureEngineering.aggregateCrashesPerGrid(spark, larimerGrid, crashes, "id");
        //Combine all of the above tables as one big attributes by grid table, joining them to road features so only the relevant grids (those with roads) are taken into account 
        Dataset<Row> allFeatures = roadFeatures;

        allFeatures = allFeatures.join(
                signFeatures,
                allFeatures.col("grid_id").equalTo(signFeatures.col("grid_id")),
                "left_outer"
        ).drop(signFeatures.col("grid_id"));
        allFeatures = allFeatures.join(
                guardrailFeatures,
                allFeatures.col("grid_id").equalTo(guardrailFeatures.col("grid_id")),
                "left_outer"
        ).drop(guardrailFeatures.col("grid_id"));
        allFeatures = allFeatures.join(
                medianFeatures,
                allFeatures.col("grid_id").equalTo(medianFeatures.col("grid_id")),
                "left_outer"
        ).drop(medianFeatures.col("grid_id"));
        allFeatures = allFeatures.join(
                sidewalkFeatures,
                allFeatures.col("grid_id").equalTo(sidewalkFeatures.col("grid_id")),
                "left_outer"
        ).drop(sidewalkFeatures.col("grid_id"));
        allFeatures = allFeatures.join(
                crashFeatures,
                allFeatures.col("grid_id").equalTo(crashFeatures.col("grid_id")),
                "left_outer"
        ).drop(crashFeatures.col("grid_id"));
        allFeatures = allFeatures.na().fill(0);

        //System.out.println("\n=== All features per grid cell that contains road ===");
        //fullDataset.show(10, false);
        allFeatures.groupBy().agg(count("*").alias("rows"), sum(col("crash_count").isNull().cast("int")).alias("null_crashes"), sum((col("crash_count").equalTo(0)).cast("int")).alias("zero_crashes")).show(false);

        // Train the model
        Dataset<Row> rfPredictions = Modeling.randomForestTrain(allFeatures);
        // We only really need grid_id + prediction to join back
        Dataset<Row> predForJoin = rfPredictions.select("grid_id", "prediction");
        // Join predictions onto the original Larimer grid by id/grid_id
        Dataset<Row> larimerWithPred = larimerGrid.join(predForJoin, larimerGrid.col("id").equalTo(predForJoin.col("grid_id")), "left_outer").drop(predForJoin.col("grid_id"));

        System.out.println("\n=== Larimer grid with RF prediction column ===");
        larimerWithPred.select("id", "prediction").show(20, false);

        // HUZZAH! WE CAN WRITE THE OUTPUT!
        larimerWithPred.write()
                .mode("overwrite")
                .parquet("output/LarimerGrid_with_rf_prediction.parquet");

        spark.stop();
        System.out.println("\n=== Done ===");
    }

    /**
     * Load a Parquet file and parse the geometry
     */
    private static Dataset<Row> loadParquetWithGeom(SparkSession spark, String path) {
        Dataset<Row> df = spark.read().parquet(path);

        String[] cols = df.columns();
        boolean hasGeom = Arrays.asList(cols).contains("geom");
        boolean hasGeometry = Arrays.asList(cols).contains("geometry");

        if (!hasGeom) {
            if (hasGeometry) {
                df = df.withColumn("geom", expr("ST_GeomFromWKB(geometry)"));
            } else {
                throw new IllegalStateException(
                        "Parquet file " + path + " has no `geom` or `geometry` column."
                );
            }
        }

        return df;
    }

    private static void printSchemaAndCount(String label, Dataset<Row> df) {
        System.out.println("\n=== " + label + " schema ===");
        df.printSchema();
        System.out.println(label + " count: " + df.count());
    }
}
