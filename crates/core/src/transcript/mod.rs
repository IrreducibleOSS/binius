// Copyright 2024 Irreducible Inc.

mod error;

use crate::{fiat_shamir::Challenger, merkle_tree::MerkleCap};
use binius_field::{deserialize_canonical, serialize_canonical, PackedField, TowerField};
use bytes::{buf::UninitSlice, Buf, BufMut, Bytes, BytesMut};
pub use error::Error;
use p3_challenger::{CanObserve, CanSample, CanSampleBits};
use std::slice;
use tracing::warn;

/// Writable(Prover) transcript over some Challenger that `CanWrite` and `CanSample<F: TowerField>`
///
/// A Transcript is an abstraction over Fiat-Shamir so the prover and verifier can send and receive
/// data, everything that gets written to or read from the transcript will be observed
#[derive(Debug, Default)]
pub struct TranscriptWriter<Challenger> {
	combined: FiatShamirBuf<BytesMut, Challenger>,
}

/// Writable(Prover) advice that `CanWrite`
///
/// Advice holds meta-data to the transcript that need not be Fiat-Shamir'ed
#[derive(Debug, Default)]
pub struct AdviceWriter {
	buffer: BytesMut,
}

/// Readable(Verifier) transcript over some Challenger that `CanRead` and `CanSample<F: TowerField>`
///
/// You must manually call the destructor with `finalize()` to check anything that's written is
/// fully read out
#[derive(Debug)]
pub struct TranscriptReader<Challenger> {
	combined: FiatShamirBuf<Bytes, Challenger>,
}

/// Readable(Verifier) advice that `CanRead`
///
/// You must manually call the destructor with `finalize()` to check anything that's written is
/// fully read out
#[derive(Debug)]
pub struct AdviceReader {
	buffer: Bytes,
}

/// Helper struct combining Transcript and Advice data to create a Proof object
pub struct Proof<Transcript, Advice> {
	pub transcript: Transcript,
	pub advice: Advice,
}

#[derive(Debug, Default)]
struct FiatShamirBuf<Inner, Challenger> {
	buffer: Inner,
	challenger: Challenger,
}

impl<Inner: Buf, Challenger_: Challenger> Buf for FiatShamirBuf<Inner, Challenger_> {
	fn remaining(&self) -> usize {
		self.buffer.remaining()
	}

	fn chunk(&self) -> &[u8] {
		self.buffer.chunk()
	}

	fn advance(&mut self, cnt: usize) {
		assert!(cnt <= self.buffer.remaining());
		// Get the slice that was written to the inner buf, observe that and advance
		let readable = self.buffer.chunk();
		// Because our internal buffer is created from vec, this should never happen.
		assert!(cnt <= readable.len());
		self.challenger.observer().put_slice(&readable[..cnt]);
		self.buffer.advance(cnt);
	}
}

unsafe impl<Inner: BufMut, Challenger_: Challenger> BufMut for FiatShamirBuf<Inner, Challenger_> {
	fn remaining_mut(&self) -> usize {
		self.buffer.remaining_mut()
	}

	unsafe fn advance_mut(&mut self, cnt: usize) {
		assert!(cnt <= self.buffer.remaining_mut());
		let written = self.buffer.chunk_mut();
		// Because out internal buffer is BytesMut cnt <= written.len(), but adding as per implementation notes
		assert!(cnt <= written.len());

		// NOTE: This is the unsafe part, you are reading the next cnt bytes on the assumption that
		// caller is ensured us the next cnt bytes are initialized.
		let written: &[u8] = slice::from_raw_parts(written.as_mut_ptr(), cnt);

		self.challenger.observer().put_slice(written);
		self.buffer.advance_mut(cnt);
	}

	fn chunk_mut(&mut self) -> &mut UninitSlice {
		self.buffer.chunk_mut()
	}
}

impl<Challenger: Default> TranscriptWriter<Challenger> {
	pub fn new() -> Self {
		Self::default()
	}

	pub fn finalize(self) -> Vec<u8> {
		self.combined.buffer.to_vec()
	}

	pub fn into_reader(self) -> TranscriptReader<Challenger> {
		TranscriptReader::new(self.finalize())
	}
}

impl AdviceWriter {
	pub fn new() -> Self {
		Self::default()
	}

	pub fn finalize(self) -> Vec<u8> {
		self.buffer.to_vec()
	}

	pub fn into_reader(self) -> AdviceReader {
		AdviceReader::new(self.finalize())
	}
}

impl<Challenger: Default> Proof<TranscriptWriter<Challenger>, AdviceWriter> {
	pub fn into_verifier(self) -> Proof<TranscriptReader<Challenger>, AdviceReader> {
		Proof {
			transcript: self.transcript.into_reader(),
			advice: self.advice.into_reader(),
		}
	}
}

impl<Challenger: Default> TranscriptReader<Challenger> {
	pub fn new(vec: Vec<u8>) -> Self {
		Self {
			combined: FiatShamirBuf {
				challenger: Challenger::default(),
				buffer: Bytes::from(vec),
			},
		}
	}

	pub fn finalize(self) -> Result<(), Error> {
		if self.combined.buffer.has_remaining() {
			return Err(Error::TranscriptNotEmpty {
				remaining: self.combined.buffer.remaining(),
			});
		}
		Ok(())
	}
}

// Useful warnings to see if we are neglecting to read any advice or transcript entirely
impl<Challenger> Drop for TranscriptReader<Challenger> {
	fn drop(&mut self) {
		if self.combined.buffer.has_remaining() {
			warn!(
				"Transcript reader is not fully read out: {:?} bytes left",
				self.combined.buffer.remaining()
			)
		}
	}
}

impl Drop for AdviceReader {
	fn drop(&mut self) {
		if self.buffer.has_remaining() {
			warn!("Advice reader is not fully read out: {:?} bytes left", self.buffer.remaining())
		}
	}
}

impl AdviceReader {
	pub fn new(vec: Vec<u8>) -> Self {
		Self {
			buffer: Bytes::from(vec),
		}
	}

	pub fn finalize(self) -> Result<(), Error> {
		if self.buffer.has_remaining() {
			return Err(Error::TranscriptNotEmpty {
				remaining: self.buffer.remaining(),
			});
		}
		Ok(())
	}
}

/// Trait that is used to read bytes and field elements from transcript/advice
#[auto_impl::auto_impl(&mut)]
pub trait CanRead {
	fn read_bytes(&mut self, buf: &mut [u8]) -> Result<(), Error>;

	fn read_scalar<F: TowerField>(&mut self) -> Result<F, Error> {
		let mut out = F::default();
		self.read_scalar_slice_into(slice::from_mut(&mut out))?;
		Ok(out)
	}

	fn read_scalar_slice_into<F: TowerField>(&mut self, buf: &mut [F]) -> Result<(), Error>;

	fn read_scalar_slice<F: TowerField>(&mut self, len: usize) -> Result<Vec<F>, Error> {
		let mut elems = vec![F::default(); len];
		self.read_scalar_slice_into(&mut elems)?;
		Ok(elems)
	}

	fn read_packed<P: PackedField<Scalar: TowerField>>(&mut self) -> Result<P, Error> {
		let mut pack = P::default();

		for i in 0..P::WIDTH {
			let x = self.read_scalar()?;
			unsafe {
				pack.set_unchecked(i, x);
			};
		}

		Ok(pack)
	}
}

impl<Challenger_: Challenger> CanRead for TranscriptReader<Challenger_> {
	fn read_bytes(&mut self, buf: &mut [u8]) -> Result<(), Error> {
		if self.combined.buffer.remaining() < buf.len() {
			return Err(Error::NotEnoughBytes);
		}
		self.combined.copy_to_slice(buf);
		Ok(())
	}

	fn read_scalar_slice_into<F: TowerField>(&mut self, buf: &mut [F]) -> Result<(), Error> {
		for elem in buf {
			*elem = deserialize_canonical(&mut self.combined)?;
		}
		Ok(())
	}
}

impl CanRead for AdviceReader {
	fn read_bytes(&mut self, buf: &mut [u8]) -> Result<(), Error> {
		if self.buffer.remaining() < buf.len() {
			return Err(Error::NotEnoughBytes);
		}
		self.buffer.copy_to_slice(buf);
		Ok(())
	}

	fn read_scalar_slice_into<F: TowerField>(&mut self, buf: &mut [F]) -> Result<(), Error> {
		for elem in buf {
			*elem = deserialize_canonical(&mut self.buffer)?;
		}
		Ok(())
	}
}

/// Trait that is used to write bytes and field elements to transcript/advice
#[auto_impl::auto_impl(&mut)]
pub trait CanWrite {
	fn write_bytes(&mut self, data: &[u8]);

	fn write_scalar<F: TowerField>(&mut self, f: F) {
		self.write_scalar_slice(slice::from_ref(&f));
	}

	fn write_scalar_slice<F: TowerField>(&mut self, elems: &[F]);

	fn write_packed<P: PackedField<Scalar: TowerField>>(&mut self, packed: P) {
		for scalar in packed.iter() {
			self.write_scalar(scalar);
		}
	}
}

impl<Challenger_: Challenger> CanWrite for TranscriptWriter<Challenger_> {
	fn write_bytes(&mut self, data: &[u8]) {
		self.combined.put_slice(data);
	}

	fn write_scalar_slice<F: TowerField>(&mut self, elems: &[F]) {
		for elem in elems {
			serialize_canonical(*elem, &mut self.combined).unwrap();
		}
	}
}

impl CanWrite for AdviceWriter {
	fn write_bytes(&mut self, data: &[u8]) {
		self.buffer.put_slice(data);
	}

	fn write_scalar_slice<F: TowerField>(&mut self, elems: &[F]) {
		for elem in elems {
			serialize_canonical(*elem, &mut self.buffer).expect("Buffer full");
		}
	}
}

impl<F, Challenger_> CanSample<F> for TranscriptReader<Challenger_>
where
	F: TowerField,
	Challenger_: Challenger,
{
	fn sample(&mut self) -> F {
		deserialize_canonical(self.combined.challenger.sampler())
			.expect("challenger has infinite buffer")
	}
}

impl<F, Challenger_> CanSample<F> for TranscriptWriter<Challenger_>
where
	F: TowerField,
	Challenger_: Challenger,
{
	fn sample(&mut self) -> F {
		deserialize_canonical(self.combined.challenger.sampler())
			.expect("challenger has infinite buffer")
	}
}

fn sample_bits_reader<Reader: Buf>(mut reader: Reader, bits: usize) -> usize {
	let bits = bits.min(usize::BITS as usize);

	let bytes_to_sample = bits.div_ceil(8);

	let mut bytes = [0u8; std::mem::size_of::<usize>()];

	reader.copy_to_slice(&mut bytes[..bytes_to_sample]);

	let unmasked = usize::from_le_bytes(bytes);
	let mask = 1usize.checked_shl(bits as u32);
	let mask = match mask {
		Some(x) => x - 1,
		None => usize::MAX,
	};
	mask & unmasked
}

impl<Challenger_> CanSampleBits<usize> for TranscriptReader<Challenger_>
where
	Challenger_: Challenger,
{
	fn sample_bits(&mut self, bits: usize) -> usize {
		sample_bits_reader(self.combined.challenger.sampler(), bits)
	}
}

impl<Challenger_> CanSampleBits<usize> for TranscriptWriter<Challenger_>
where
	Challenger_: Challenger,
{
	fn sample_bits(&mut self, bits: usize) -> usize {
		sample_bits_reader(self.combined.challenger.sampler(), bits)
	}
}

// This is temporary until we can move the entire proof into read write interface
impl<P, Challenger_> CanObserve<MerkleCap<P>> for TranscriptReader<Challenger_>
where
	Challenger_: Challenger,
	P: Clone,
	Self: CanObserve<P>,
{
	fn observe(&mut self, value: MerkleCap<P>) {
		self.observe_slice(&value.0);
	}
}

impl<P, Challenger_> CanObserve<P> for TranscriptReader<Challenger_>
where
	P: PackedField<Scalar: TowerField>,
	Challenger_: Challenger,
{
	fn observe(&mut self, value: P) {
		for scalar in value.iter() {
			serialize_canonical(scalar, self.combined.challenger.observer())
				.expect("challenger has infinite buffer")
		}
	}
}

impl<P, Challenger_> CanObserve<MerkleCap<P>> for TranscriptWriter<Challenger_>
where
	P: Clone,
	Self: CanObserve<P>,
	Challenger_: Challenger,
{
	fn observe(&mut self, value: MerkleCap<P>) {
		self.observe_slice(&value.0);
	}
}

impl<P, Challenger_> CanObserve<P> for TranscriptWriter<Challenger_>
where
	P: PackedField<Scalar: TowerField>,
	Challenger_: Challenger,
{
	fn observe(&mut self, value: P) {
		for scalar in value.iter() {
			serialize_canonical(scalar, self.combined.challenger.observer())
				.expect("challenger has infinite buffer")
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::fiat_shamir::HasherChallenger;
	use binius_field::{
		AESTowerField128b, AESTowerField16b, AESTowerField32b, AESTowerField8b, BinaryField128b,
		BinaryField128bPolyval, BinaryField32b, BinaryField64b, BinaryField8b,
	};
	use groestl_crypto::Groestl256;
	use rand::{thread_rng, RngCore};

	#[test]
	fn test_transcripting() {
		let mut prover_transcript = TranscriptWriter::<HasherChallenger<Groestl256>>::new();

		prover_transcript.write_scalar(BinaryField8b::new(0x96));
		prover_transcript.write_scalar(BinaryField32b::new(0xDEADBEEF));
		prover_transcript.write_scalar(BinaryField128b::new(0x55669900112233550000CCDDFFEEAABB));
		prover_transcript.observe(BinaryField64b::new(0xAA11223344556677));
		let sampled_fanpaar1: BinaryField128b = prover_transcript.sample();

		prover_transcript.write_scalar(AESTowerField8b::new(0x52));
		prover_transcript.write_scalar(AESTowerField32b::new(0x12345678));
		prover_transcript.write_scalar(AESTowerField128b::new(0xDDDDBBBBCCCCAAAA2222999911117777));
		prover_transcript.observe(AESTowerField8b::new(0x20));

		let sampled_aes1: AESTowerField16b = prover_transcript.sample();

		prover_transcript
			.write_scalar(BinaryField128bPolyval::new(0xFFFF12345678DDDDEEEE87654321AAAA));
		prover_transcript.observe(BinaryField128bPolyval::new(0xEEEE87654321AAAAFFFF12345678DDDD));
		let sampled_polyval1: BinaryField128bPolyval = prover_transcript.sample();

		let mut verifier_transcript = prover_transcript.into_reader();

		let fp_8: BinaryField8b = verifier_transcript.read_scalar().unwrap();
		let fp_32: BinaryField32b = verifier_transcript.read_scalar().unwrap();
		let fp_128: BinaryField128b = verifier_transcript.read_scalar().unwrap();

		assert_eq!(fp_8.val(), 0x96);
		assert_eq!(fp_32.val(), 0xDEADBEEF);
		assert_eq!(fp_128.val(), 0x55669900112233550000CCDDFFEEAABB);

		verifier_transcript.observe(BinaryField64b::new(0xAA11223344556677));
		let sampled_fanpaar1_res: BinaryField128b = verifier_transcript.sample();

		assert_eq!(sampled_fanpaar1_res, sampled_fanpaar1);

		let aes_8: AESTowerField8b = verifier_transcript.read_scalar().unwrap();
		let aes_32: AESTowerField32b = verifier_transcript.read_scalar().unwrap();
		let aes_128: AESTowerField128b = verifier_transcript.read_scalar().unwrap();

		assert_eq!(aes_8.val(), 0x52);
		assert_eq!(aes_32.val(), 0x12345678);
		assert_eq!(aes_128.val(), 0xDDDDBBBBCCCCAAAA2222999911117777);

		verifier_transcript.observe(AESTowerField8b::new(0x20));
		let sampled_aes_res: AESTowerField16b = verifier_transcript.sample();

		assert_eq!(sampled_aes_res, sampled_aes1);

		let polyval_128: BinaryField128bPolyval = verifier_transcript.read_scalar().unwrap();
		assert_eq!(polyval_128, BinaryField128bPolyval::new(0xFFFF12345678DDDDEEEE87654321AAAA));

		verifier_transcript
			.observe(BinaryField128bPolyval::new(0xEEEE87654321AAAAFFFF12345678DDDD));
		let sampled_polyval_res: BinaryField128bPolyval = verifier_transcript.sample();
		assert_eq!(sampled_polyval_res, sampled_polyval1);

		verifier_transcript.finalize().unwrap();
	}

	#[test]
	fn test_advicing() {
		let mut advice_writer = AdviceWriter::new();

		advice_writer.write_scalar(BinaryField8b::new(0x96));
		advice_writer.write_scalar(BinaryField32b::new(0xDEADBEEF));
		advice_writer.write_scalar(BinaryField128b::new(0x55669900112233550000CCDDFFEEAABB));

		advice_writer.write_scalar(AESTowerField8b::new(0x52));
		advice_writer.write_scalar(AESTowerField32b::new(0x12345678));
		advice_writer.write_scalar(AESTowerField128b::new(0xDDDDBBBBCCCCAAAA2222999911117777));

		advice_writer.write_scalar(BinaryField128bPolyval::new(0xFFFF12345678DDDDEEEE87654321AAAA));

		let mut advice_reader = advice_writer.into_reader();

		let fp_8: BinaryField8b = advice_reader.read_scalar().unwrap();
		let fp_32: BinaryField32b = advice_reader.read_scalar().unwrap();
		let fp_128: BinaryField128b = advice_reader.read_scalar().unwrap();

		assert_eq!(fp_8.val(), 0x96);
		assert_eq!(fp_32.val(), 0xDEADBEEF);
		assert_eq!(fp_128.val(), 0x55669900112233550000CCDDFFEEAABB);

		let aes_8: AESTowerField8b = advice_reader.read_scalar().unwrap();
		let aes_32: AESTowerField32b = advice_reader.read_scalar().unwrap();
		let aes_128: AESTowerField128b = advice_reader.read_scalar().unwrap();

		assert_eq!(aes_8.val(), 0x52);
		assert_eq!(aes_32.val(), 0x12345678);
		assert_eq!(aes_128.val(), 0xDDDDBBBBCCCCAAAA2222999911117777);

		let polyval_128: BinaryField128bPolyval = advice_reader.read_scalar().unwrap();
		assert_eq!(polyval_128, BinaryField128bPolyval::new(0xFFFF12345678DDDDEEEE87654321AAAA));

		advice_reader.finalize().unwrap();
	}

	#[test]
	fn test_challenger() {
		let mut transcript = TranscriptWriter::<HasherChallenger<Groestl256>>::new();
		let mut challenger = HasherChallenger::<Groestl256>::default();

		const NUM_SAMPLING: usize = 32;
		let mut random_bytes = [0u8; NUM_SAMPLING * 8];
		thread_rng().fill_bytes(&mut random_bytes);
		let mut sampled_arrays = [[0u8; 8]; NUM_SAMPLING];

		for i in 0..NUM_SAMPLING {
			transcript.write_scalar(BinaryField64b::new(u64::from_le_bytes(
				random_bytes[i * 8..i * 8 + 8].to_vec().try_into().unwrap(),
			)));
			challenger
				.observer()
				.put_slice(&random_bytes[i * 8..i * 8 + 8]);

			let sampled_out_transcript: BinaryField64b = transcript.sample();
			let mut challenger_out = [0u8; 8];
			challenger.sampler().copy_to_slice(&mut challenger_out);
			assert_eq!(challenger_out, sampled_out_transcript.val().to_le_bytes());
			sampled_arrays[i] = challenger_out;
		}

		let mut transcript = transcript.into_reader();

		for array in sampled_arrays.into_iter() {
			let _: BinaryField64b = transcript.read_scalar().unwrap();
			let sampled_out_transcript: BinaryField64b = transcript.sample();

			assert_eq!(array, sampled_out_transcript.val().to_le_bytes());
		}

		transcript.finalize().unwrap();
	}
}
