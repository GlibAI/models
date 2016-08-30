py_library(
    name = "utils",
    srcs = ["utils.py"],
    deps = [
        "//syntaxnet:sentence_py_pb2",
        "//syntaxnet:task_spec_py_pb2",
    ],
)

py_library(
    name = "tokenizer",
    srcs = ["tokenizer.py"],
    deps = [
        "//syntaxnet:graph_builder",
        "//syntaxnet:structured_graph_builder",
        ":utils",
    ],
)

py_library(
    name = "tagger",
    srcs = ["tagger.py"],
    deps = [
        "//syntaxnet:graph_builder",
        "//syntaxnet:structured_graph_builder",
        ":utils",
    ],
)

py_library(
    name = "parser",
    srcs = ["parser.py"],
    deps = [
        "//syntaxnet:graph_builder",
        "//syntaxnet:structured_graph_builder",
        ":utils",
    ],
)

py_library(
    name = "glib",
    srcs = ["glib.py"],
    deps = [
        ":utils",
        ":tokenizer",
        ":tagger",
        ":parser",
    ],
)

py_binary(
    name = "server",
    srcs = ["server.py"],
    deps = [
        ":glib",
    ],
)