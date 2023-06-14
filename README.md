
# Medical Overseer Control (MOC) ML API

The MOC ML API is a ML serving API that is interconnected with our main API. It uses FastAPI framework for building the API and a tensorflow based ML model for extracting the estimated data about how many people are in the room from a JPEG/PNG image. Our ML API are hosted on the GCP App engine's platform, making it easy to deploy and hassle-free




## Authors

- [Doni Febrian](https://www.github.com/peepeeyanto)
- [Alvan Alfiansyah](https://www.github.com/alvansoleh)


## Deployment

To deploy this project you can clone this project through GCP's cloud shell, and then deploy it easily through GCP's App Engine using

```bash
  gcloud app deploy
```
Note that you can edit the app.yaml file to change the instance type and number of workers. **IMPORTANT** For one worker, this app requires about 700mb of memory or greater. So for one worker you'll need a minimal of F2 instance or greater


## API Reference

#### Process image
This endpoint is for extracting the estimated number of people in the room from a JPEG/PNG image

```http
  POST /predict
```

| Fields | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `file` | `JPEG/PNG Image` | **Required**. Your image that need to be processed |


