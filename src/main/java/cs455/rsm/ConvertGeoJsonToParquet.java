package cs455.rsm;

import org.apache.sedona.sql.utils.SedonaSQLRegistrator;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import static org.apache.spark.sql.functions.col;
import static org.apache.spark.sql.functions.explode;
import static org.apache.spark.sql.functions.expr;
import static org.apache.spark.sql.functions.to_json;
public class ConvertGeoJsonToParquet {

    public static void main(String[] args) {

        if (args.length != 2) {
            System.err.println("Usage: ConvertGeoJsonToParquet <input.geojson> <output.parquet>");
            System.exit(1);
        }

        String inputPath  = args[0];
        String outputPath = args[1];

        SparkSession spark = SparkSession.builder()
                .appName("ConvertGeoJsonToParquet")
                .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
                .config("spark.kryo.registrator",
                        "org.apache.sedona.core.serde.SedonaKryoRegistrator")
                .getOrCreate();

        spark.sparkContext().setLogLevel("WARN");
        SedonaSQLRegistrator.registerAll(spark);

        System.out.println("\n=== Reading GeoJSON: " + inputPath + " ===");

        Dataset<Row> raw = spark.read()
                .option("multiLine", "true")   // QGIS-style FeatureCollection
                .json(inputPath);

        // Explode FeatureCollection (one row per feature)
        Dataset<Row> df = raw
                .select(explode(col("features")).as("feature"))
                .select(
                        col("feature.properties.*"),               // attributes
                        to_json(col("feature.geometry")).as("geom_json")
                )
                .withColumn("geom", expr("ST_GeomFromGeoJSON(geom_json)"))
                .drop("geom_json");

        System.out.println("\n=== Parsed schema ===");
        df.printSchema();

        System.out.println("Feature count = " + df.count());

        // Write Parquet (columnar, compressed, fast!)
        df.write()
                .mode("overwrite")
                .parquet(outputPath);

        System.out.println("\n=== Wrote Parquet to: " + outputPath + " ===");

        spark.stop();
    }
}