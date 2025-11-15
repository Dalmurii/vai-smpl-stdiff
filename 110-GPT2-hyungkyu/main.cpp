void tokenizerTest();
void dataLoaderTest();
void embeddingNodeTest();
void attentionNodeTest();

int main()
{
    // Run tokenizer tests
    tokenizerTest();

    // Run dataloader tests
    dataLoaderTest();

    // Run embedding node tests (Vulkan version)
    embeddingNodeTest();

    // Run attention node tests (Multi-Head Attention)
    attentionNodeTest();

    return 0;
}

