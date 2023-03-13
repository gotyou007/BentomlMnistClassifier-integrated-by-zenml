from zenml.pipelines import pipeline

@pipeline(enable_cache=False)
def deploy_fashion_mnist(
    importer,
    model_loader,
    evaluator,
    deployment_trigger,
    bento_builder,
    deployer,
):
    _, test_dataloader = importer()
    model = model_loader()
    accuracy = evaluator(test_dataloader=test_dataloader, model=model)
    decision = deployment_trigger(accuracy=accuracy)
    bento = bento_builder(model=model)
    deployer(deploy_decision=decision, bento=bento)
