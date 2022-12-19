import { TestBed } from '@angular/core/testing';

import { SendJpegService } from './send-jpeg.service';

describe('SendJpegService', () => {
  let service: SendJpegService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(SendJpegService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
