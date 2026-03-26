# Config
SPARK_HOME      ?= C:/spark/spark-4.1.1-bin-hadoop3
SRC_DIR         := src/main/java
BUILD_DIR       := build
JAR_NAME        := rsm-1.0-SNAPSHOT.jar
MAIN_CLASS      := rsm.Main
SEDONA_PACKAGES := org.apache.sedona:sedona-spark-shaded-3.5_2.13:1.8.1,org.datasyslab:geotools-wrapper:1.8.1-33.1
JAR_CMD := $(JAVA_HOME)/bin/jar

.PHONY: all compile jar run test clean

all: jar

compile:
	if not exist $(BUILD_DIR)\classes mkdir $(BUILD_DIR)\classes
	javac \
		-cp "$(SPARK_HOME)/jars/*;lib/*" \
		-d $(BUILD_DIR)/classes \
		$(SRC_DIR)/rsm/ConvertGeoJsonToParquet.java \
		$(SRC_DIR)/rsm/FeatureEngineering.java \
		$(SRC_DIR)/rsm/Main.java \
		$(SRC_DIR)/rsm/Modeling.java

jar: compile
	"$(JAR_CMD)" cf $(BUILD_DIR)/$(JAR_NAME) -C $(BUILD_DIR)/classes .

run: jar
	spark-submit \
		--class $(MAIN_CLASS) \
		--packages $(SEDONA_PACKAGES) \
		$(BUILD_DIR)/$(JAR_NAME)

test: jar
	spark-submit \
		--class $(MAIN_CLASS) \
		--packages $(SEDONA_PACKAGES) \
		$(BUILD_DIR)/$(JAR_NAME) \
		--test

clean:
	if exist $(BUILD_DIR) rmdir /s /q $(BUILD_DIR)