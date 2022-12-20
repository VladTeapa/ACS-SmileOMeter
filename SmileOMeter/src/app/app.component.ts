import { Component, OnInit } from '@angular/core';
import { Observable, Subject } from 'rxjs';
import { WebcamImage, WebcamInitError, WebcamUtil } from 'ngx-webcam';
import { SendJpegService } from './send-jpeg.service';
import { DomSanitizer } from '@angular/platform-browser';
import { JpegResponse } from './model/jpeg-response';
@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent implements OnInit{
  title = 'SmileOMeter';
  private trigger: Subject<any> = new Subject();
  public webcamImage!: WebcamImage;
  private nextWebcam: Subject<any> = new Subject();
  private sendJpegService: SendJpegService;
  public angry: number=0;
  public sad: number=0;
  public neutral: number=0;
  public surprise: number=0;
  public fear: number=0;
  public happy: number=0;
  public disgust: number=0;
  public imagePath: String;
  sysImage = '';
  constructor(service: SendJpegService, private _sanitizer: DomSanitizer){
    this.sendJpegService = service;
    this.imagePath = "";
  }
  ngOnInit() {
  }
  public getSnapshot(): void {
    this.trigger.next(void 0);
  }
  public captureImg(webcamImage: WebcamImage): void {
    this.webcamImage = webcamImage;
    var arr = this.webcamImage.imageAsDataUrl.split(",");
    var mime;
    if(arr[0]!=null)
      mime = arr[0].match(/:(.*?);/);
    else
      mime = "";
    if(mime != null && mime[1]!=null)
    {
       mime = mime[1];
    }
    const bstr = atob(arr[1]);
    let n = bstr.length;
    const u8arr = new Uint8Array(n);
    while (n--) {
      u8arr[n] = bstr.charCodeAt(n);
    }
    const file: File = new File([u8arr], "image", { type: "jpeg" })
    this.webcamImage = webcamImage;
    this.sysImage = webcamImage!.imageAsDataUrl;
    this.sendJpegService.getImage(file).subscribe(
      (response: JpegResponse) => {
        console.info(response)
        this.imagePath = new String('data:image/jpg;base64,' 
                 + response['data']);
        this.angry = response['emotions']['angry']
        this.sad = response['emotions']['sad']
        this.disgust = response['emotions']['disgust']
        this.fear = response['emotions']['fear']
        this.happy = response['emotions']['happy']
        this.surprise = response['emotions']['surprise']
        this.neutral = response['emotions']['neutral']
        this.getSnapshot();
      },
      (error: any) => {
        console.info(error['error']['text'])
        this.getSnapshot();
      }
    );
    console.info('got webcam image', this.sysImage);
    //subscribe
  }
  public get invokeObservable(): Observable<any> {
    return this.trigger.asObservable();
  }
  public get nextWebcamObservable(): Observable<any> {
    return this.nextWebcam.asObservable();
  }
}
