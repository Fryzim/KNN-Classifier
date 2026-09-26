// Power Query (M) — Waveform dataset (k-NN project)
//
// Usage: Power BI Desktop > Get Data > Blank Query > Advanced Editor > paste this,
// then set FilePath below to the local path of waveform.data.csv.
//
// The raw file has no header row: 21 continuous features + 1 integer class
// label (0/1/2). This script names the columns and types them so the report
// can filter/aggregate by class directly.

let
    FilePath = "C:\Path\To\KNN-Classifier\waveform.data.csv",

    Source = Csv.Document(
        File.Contents(FilePath),
        [Delimiter=",", Columns=22, Encoding=65001, QuoteStyle=QuoteStyle.None]
    ),

    FeatureNames = List.Transform({1..21}, each "feature_" & Text.From(_)),
    ColumnNames = FeatureNames & {"class"},
    Renamed = Table.RenameColumns(Source, List.Zip({Table.ColumnNames(Source), ColumnNames})),

    TypedFeatures = List.Transform(FeatureNames, each {_, type number}),
    Typed = Table.TransformColumnTypes(Renamed, TypedFeatures & {{"class", Int64.Type}}),

    AddClassLabel = Table.AddColumn(Typed, "class_label", each "Classe " & Text.From([class]), type text)
in
    AddClassLabel
