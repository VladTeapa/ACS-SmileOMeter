import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders, HttpParams } from '@angular/common/http';
import { Observable } from 'rxjs';
import { JpegResponse } from './model/jpeg-response';
@Injectable({
  providedIn: 'root'
})
export class SendJpegService {
  private apiServerUrl = "http://localhost:44444/";
  private httpOptions = {
    headers: new HttpHeaders({
      'Content-Type':  'application/json',
      Authorization: 'Bearer ',
      'Access-Control-Allow-Origin': '*'
 
    },
    )
  };
  constructor(private http: HttpClient) { }

  public getImage(file: File): Observable <JpegResponse> {
    const formData = new FormData();
    formData.append("file", file);
    return this.http.post<JpegResponse>(`${this.apiServerUrl}`, formData);
  }

}
