gradle clean build
spark-submit \
  --class cs455.rsm.Main \
  --packages org.apache.sedona:sedona-core-3.0_2.12:1.4.1,org.apache.sedona:sedona-sql-3.0_2.12:1.4.1 \
  build/libs/rsm-1.0-SNAPSHOT.jar
