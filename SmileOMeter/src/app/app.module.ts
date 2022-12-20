import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { HttpClientModule } from '@angular/common/http';
import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { WebcamModule } from 'ngx-webcam';
import { SendJpegService } from './send-jpeg.service';
@NgModule({
  declarations: [
    AppComponent
  ],
  imports: [
    BrowserModule,
    AppRoutingModule,
    WebcamModule,    
    HttpClientModule
  ],
  providers: [SendJpegService],
  bootstrap: [AppComponent]
})
export class AppModule { }
