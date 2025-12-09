package cs455.rsm;

import org.apache.spark.sql.Column;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import static org.apache.spark.sql.functions.col;
import static org.apache.spark.sql.functions.lit;
import static org.apache.spark.sql.functions.not;
import static org.apache.spark.sql.functions.regexp_replace;
import static org.apache.spark.sql.functions.sum;
import static org.apache.spark.sql.functions.trim;
import static org.apache.spark.sql.functions.when;

public class FeatureEngineering {

    /**
     * Aggregate sign panel counts per grid cell into a small number of
     * safety-relevant categories (sign_group), then pivot. Signs schema (Sign
     * Panels / Mounts): MUTCDDESCD (string) - descriptive MUTCD code OBJECTID
     * (long) geom (geometry) - point geometry
     */
    public static Dataset<Row> aggregateSignsFullPivot(
            SparkSession spark,
            Dataset<Row> grid,
            Dataset<Row> signs,
            String gridIdCol,
            String descCol
    ) {
        grid.createOrReplaceTempView("grid");
        signs.createOrReplaceTempView("signs");

        // Spatial join: each sign gets assigned to a grid cell (if contained)
        Dataset<Row> joined = spark.sql(
                "SELECT g." + gridIdCol + " AS grid_id, "
                + "       s." + descCol + " AS sign_type "
                + "FROM grid g "
                + "LEFT JOIN signs s ON ST_Contains(g.geom, s.geom)"
        );

        // Basic cleaning: drop null/blank/obvious junk
        Dataset<Row> cleaned = joined
                .filter(col("sign_type").isNotNull())
                .filter(trim(col("sign_type")).notEqual(""))
                .filter(not(col("sign_type").equalTo("_")))
                .filter(not(col("sign_type").equalTo("null")))
                .filter(not(col("sign_type").equalTo("999___MISC._OTHER_SIGN")))
                .filter(not(col("sign_type").equalTo("R_999___MISC._REGULATORY_SIGNS")))
                .filter(not(col("sign_type").equalTo("W_999___MISC._WARNING_SIGNS")))
                .filter(not(col("sign_type").equalTo("INFO___GENERAL_INFORMATION_SIGNS")))
                .filter(not(col("sign_type").equalTo("REC___RECREATIONAL_AND_CULTURAL_INTEREST_AREA_SIGNS")))
                .filter(not(col("sign_type").equalTo("SERVICES___GENERAL_SERVICE_SIGNS")));

        Dataset<Row> normalized = cleaned.withColumn("sign_code", regexp_replace(col("sign_type"), "-", "_"));
        //  Map detailed MUTCD codes broad safety-relevant categories
        //   This was positively miserable to look through, turns out road signs aren't my passion
        //   speed_limit: R2_* (speed limit), school speed assemblies S4_*, S5_*
        //   stop_yield: R1_* (STOP/YIELD family)
        //   lane_row: R3_*, R4_* (turn restrictions, keep right, do not pass, etc.)
        //   curve_turn: W1_*, W13_* (alignment & advisory speed)
        //   intersection_warn: W2_* (side/cross/T/Y intersection)
        //   advance_cond: W3_* (STOP AHEAD, YIELD AHEAD, SIGNAL AHEAD, etc.)
        //   grade_surface: W7_*, W8_* (hills, grades, pavement ends, soft shoulder, etc.)
        //   ped_bike: W11_* (bike/ped/animal crossings), R9_50
        //   school_zone: S1_1 (school symbol)
        //   object_marker: OM* and W1_8 (chevrons)
        //   railroad: W10_1, R15_1
        Column signGroupCol
                = // Speed-related / school speed control
                when(col("sign_code").rlike("^R2_.*"), lit("speed_limit")) // all R2-xx*
                        .when(col("sign_code").rlike("^S4_.*|^S5_.*"), lit("school_speed"))
                        // STOP / YIELD / right-of-way control
                        .when(col("sign_code").rlike("^R1_.*"), lit("stop_yield"))
                        // Lane usage / ROW rules (turn only, keep right, do not pass, etc.)
                        .when(col("sign_code").rlike("^R3_.*|^R4_.*"), lit("lane_row_control"))
                        // Curves, turns, alignment + advisory speed plaques
                        .when(col("sign_code").rlike("^W1_.*|^W13_.*"), lit("curve_turn"))
                        // Intersection warnings (side road, cross road, T, Y, etc.)
                        .when(col("sign_code").rlike("^W2_.*"), lit("intersection_warning"))
                        // Advance condition warnings (STOP AHEAD, YIELD AHEAD, SIGNAL AHEAD, SPEED CHANGE AHEAD)
                        .when(col("sign_code").rlike("^W3_.*"), lit("advance_condition"))
                        // Vertical alignment & surface condition (grades, hills, bumps, dips, pavement ends, etc.)
                        .when(col("sign_code").rlike("^W7_.*|^W8_.*"), lit("grade_surface"))
                        // Ped/bike/animal crossings & related
                        .when(col("sign_code").rlike("^W11_.*"), lit("ped_bike_animal"))
                        .when(col("sign_code").rlike("^R9_50_.*"), lit("ped_bike_animal"))
                        // School zone presence (separate from speed enforcement if you want)
                        .when(col("sign_code").rlike("^S1_1_.*"), lit("school_zone"))
                        // Object markers & chevrons
                        .when(col("sign_code").rlike("^OM.*"), lit("object_marker"))
                        .when(col("sign_code").equalTo("W1_8___CHEVRON_ALIGNMENT"), lit("object_marker"))
                        // Railroad
                        .when(col("sign_code").rlike("^W10_.*"), lit("railroad"))
                        .when(col("sign_code").rlike("^R15_1_.*"), lit("railroad"));

        Dataset<Row> withGroups = normalized.withColumn("sign_group", signGroupCol);

        Dataset<Row> groupedRelevant = withGroups.filter(col("sign_group").isNotNull());

        Dataset<Row> groupedCounts = groupedRelevant.groupBy("grid_id", "sign_group").count();

        Dataset<Row> pivoted = groupedCounts.groupBy("grid_id").pivot("sign_group").sum("count");

        pivoted = pivoted.na().fill(0);

        return pivoted;
    }

    /**
     * Aggregate total road length per grid cell, broken out by posted speed
     * (SPEED).
     *
     * Roads schema: SPEED (int) - posted speed limit geom (geom) - line
     * geometry
     */
    public static Dataset<Row> aggregateRoadLengthsBySpeed(SparkSession spark, Dataset<Row> grid, Dataset<Row> roads, String gridIdCol) {
        // Register temp views
        grid.createOrReplaceTempView("grid");
        roads.createOrReplaceTempView("roads");

        // Spatial join: intersect roads with grids, compute length of the portion inside each grid
        Dataset<Row> joined = spark.sql(
                "SELECT g." + gridIdCol + " AS grid_id, "
                + "       r.SPEED AS speed, "
                + "       ST_Length(ST_Intersection(g.geom, r.geom)) AS seg_len "
                + "FROM grid g "
                + "LEFT JOIN roads r ON ST_Intersects(g.geom, r.geom)"
        );

        // Keep only valid speeds and positive lengths
        Dataset<Row> cleaned = joined
                .filter(col("speed").isNotNull())
                .filter(col("speed").gt(0))
                .filter(col("seg_len").gt(0));

        // Cast speed to string for pivot
        Dataset<Row> withSpeedStr = cleaned.withColumn(
                "speed_str",
                col("speed").cast("string")
        );

        // Sum length per (grid_id, speed_str)
        Dataset<Row> grouped = withSpeedStr
                .groupBy("grid_id", "speed_str")
                .agg(sum("seg_len").alias("total_length"));

        // Pivot: one column per speed
        Dataset<Row> pivoted = grouped
                .groupBy("grid_id")
                .pivot("speed_str")
                .sum("total_length");

        // Fill nulls with 0.0
        pivoted = pivoted.na().fill(0.0);

        // Clean column names: "25" -> "road_len_mph_25"
        for (String colName : pivoted.columns()) {
            if (!"grid_id".equals(colName)) {
                String safe = "road_len_mph_" + colName
                        .replace(" ", "_")
                        .replace(".", "_")
                        .replace("-", "_");
                if (!safe.equals(colName)) {
                    pivoted = pivoted.withColumnRenamed(colName, safe);
                }
            }
        }
        //Not going to add ALL of the grid cells back in here like was done for the signs, since we don't care about grid cells with no roads in them
        return pivoted;
    }

    public static Dataset<Row> aggregateGuardrailLengthsByMaterial(
            SparkSession spark,
            Dataset<Row> grid,
            Dataset<Row> guardrails,
            String gridIdCol
    ) {
        // Register temp views
        grid.createOrReplaceTempView("grid");
        guardrails.createOrReplaceTempView("guardrails");

        // Spatial join: guardrails ∩ grid, measure segment length inside each grid
        Dataset<Row> joined = spark.sql(
                "SELECT g." + gridIdCol + " AS grid_id, "
                + "       gr.MATERIALD AS material, "
                + "       ST_Length(ST_Intersection(g.geom, gr.geom)) AS seg_len "
                + "FROM grid g "
                + "LEFT JOIN guardrails gr ON ST_Intersects(g.geom, gr.geom)"
        );

        // Keep only valid materials and positive lengths
        Dataset<Row> cleaned = joined
                .filter(col("material").isNotNull())
                .filter(col("material").notEqual(""))
                .filter(col("seg_len").gt(0));

        // Sum length per (grid_id, material)
        Dataset<Row> grouped = cleaned
                .groupBy("grid_id", "material")
                .agg(sum("seg_len").alias("total_length"));

        // Pivot: one column per material type
        Dataset<Row> pivoted = grouped
                .groupBy("grid_id")
                .pivot("material")
                .sum("total_length")
                .na().fill(0.0);   // grids that have some guardrails but not all materials

        // Clean column names: "STEEL BEAM" -> "guardrail_len_STEEL_BEAM"
        for (String colName : pivoted.columns()) {
            if (!"grid_id".equals(colName)) {
                String safe = "guardrail_len_" + colName
                        .replace(" ", "_")
                        .replace(".", "_")
                        .replace("-", "_")
                        .replace("/", "_");
                if (!safe.equals(colName)) {
                    pivoted = pivoted.withColumnRenamed(colName, safe);
                }
            }
        }
        // ensure all grid cells are present.
        Dataset<Row> gridIds = grid.select(col(gridIdCol).alias("grid_id")).distinct();
        Dataset<Row> fullFeatureTable = gridIds.join(pivoted, gridIds.col("grid_id").equalTo(pivoted.col("grid_id")), "left_outer").drop(pivoted.col("grid_id")).na().fill(0.0);
        return fullFeatureTable;
    }

    public static Dataset<Row> aggregateMedianLengthsByType(SparkSession spark, Dataset<Row> grid, Dataset<Row> medians, String gridIdCol) {
        grid.createOrReplaceTempView("grid");
        medians.createOrReplaceTempView("medians");

        Dataset<Row> joined = spark.sql(
                "SELECT g." + gridIdCol + " AS grid_id, "
                + "       m.TYPED AS median_type, "
                + "       ST_Length(ST_Intersection(g.geom, m.geom)) AS seg_len "
                + "FROM grid g "
                + "LEFT JOIN medians m ON ST_Intersects(g.geom, m.geom)"
        );

        Dataset<Row> cleaned = joined
                .filter(col("median_type").isNotNull())
                .filter(col("median_type").notEqual(""))
                .filter(col("seg_len").gt(0));

        Dataset<Row> grouped = cleaned
                .groupBy("grid_id", "median_type")
                .agg(sum("seg_len").alias("total_length"));

        Dataset<Row> pivoted = grouped
                .groupBy("grid_id")
                .pivot("median_type")
                .sum("total_length")
                .na().fill(0.0);

        // rename columns to guardrail-style names
        for (String colName : pivoted.columns()) {
            if (!"grid_id".equals(colName)) {
                String safe = "median_len_" + colName
                        .replace(" ", "_")
                        .replace(".", "_")
                        .replace("-", "_")
                        .replace("/", "_");
                if (!safe.equals(colName)) {
                    pivoted = pivoted.withColumnRenamed(colName, safe);
                }
            }
        }
        // keep all grids
        Dataset<Row> gridIds = grid.select(col(gridIdCol).alias("grid_id")).distinct();
        return gridIds
                .join(pivoted,
                        gridIds.col("grid_id").equalTo(pivoted.col("grid_id")),
                        "left_outer")
                .drop(pivoted.col("grid_id"))
                .na().fill(0.0);
    }

    public static Dataset<Row> aggregateSidewalkLengths(
            SparkSession spark,
            Dataset<Row> grid,
            Dataset<Row> sidewalks,
            String gridIdCol
    ) {
        grid.createOrReplaceTempView("grid");
        sidewalks.createOrReplaceTempView("sidewalks");

        Dataset<Row> joined = spark.sql(
                "SELECT g." + gridIdCol + " AS grid_id, "
                + "       ST_Length(ST_Intersection(g.geom, s.geom)) AS seg_len "
                + "FROM grid g "
                + "LEFT JOIN sidewalks s ON ST_Intersects(g.geom, s.geom)"
        );

        Dataset<Row> cleaned = joined
                .filter(col("seg_len").gt(0));
        Dataset<Row> grouped = cleaned
                .groupBy("grid_id")
                .agg(sum("seg_len").alias("sidewalk_len"));
        // keep all grids
        Dataset<Row> gridIds = grid.select(col(gridIdCol).alias("grid_id")).distinct();
        return gridIds.join(grouped, "grid_id").na().fill(0.0);
    }

    public static Dataset<Row> aggregateCrashesPerGrid(
            SparkSession spark,
            Dataset<Row> grid,
            Dataset<Row> crashes,
            String gridIdCol
    ) {
        grid.createOrReplaceTempView("grid");
        crashes.createOrReplaceTempView("crashes");

        Dataset<Row> joined = spark.sql(
                "SELECT g." + gridIdCol + " AS grid_id, "
                + "       c.OBJECTID AS crash_id "
                + "FROM grid g "
                + "LEFT JOIN crashes c ON ST_Contains(g.geom, c.geom)"
        );

        // count only non-null crashes per grid
        Dataset<Row> grouped = joined
                .groupBy("grid_id")
                .agg(sum(col("crash_id").isNotNull().cast("int")).alias("crash_count"));

        Dataset<Row> gridIds = grid.select(col(gridIdCol).alias("grid_id")).distinct();

        return gridIds.join(grouped, gridIds.col("grid_id").equalTo(grouped.col("grid_id")), "left_outer").drop(grouped.col("grid_id")).na().fill(0);
    }

}
