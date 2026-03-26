#!/bin/bash

# Config
SPARK_HOME=${SPARK_HOME:-"C:/spark/spark-4.1.1-bin-hadoop3"}
SRC_DIR="src/main/java"
BUILD_DIR="build/"
JAR_NAME="rsm-1.0-SNAPSHOT.jar"

# Setup
mkdir -p "$BUILD_DIR/classes"

# Compile
javac \
  -cp "$SPARK_HOME/jars/*" \
  -d "$BUILD_DIR/classes" \
  $(find "$SRC_DIR" -name "*.java")

# Package
jar cf "$BUILD_DIR/$JAR_NAME" -C "$BUILD_DIR/classes" .

# Submit
spark-submit \
  --class rsm.Main \
  --packages org.apache.sedona:sedona-spark-3.5_2.13:1.7.0 \
  "$BUILD_DIR/$JAR_NAME"