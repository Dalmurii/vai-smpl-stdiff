cmake_minimum_required(VERSION 3.16)

project(110-GPT2-hyungkyu)

add_executable(${PROJECT_NAME}
    main.cpp
    tokenizer/test.cpp
    tokenizer/bpeTokenizer.cpp
    dataloader/test.cpp
    dataloader/gptDataset.cpp
    dataloader/gptDataLoader.cpp
    model/embedding/test.cpp
    model/embedding/embedding.cpp
)

add_definitions(
    -DPROJECT_CURRENT_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    nlohmann_json
)

